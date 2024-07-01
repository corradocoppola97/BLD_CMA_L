import os
import random
import numpy as np
from cmalight_BD import CMA_L_BD
from cmalight import CMA_L
#from Dataset import Dataset
import torch
from typing import Union
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torchvision.datasets as datasets


def closure(dataset: torch.utils.data.DataLoader,
            mod: torchvision.models,
            loss_fun: torch.nn,
            device: Union[torch.device, str]):
    mod.eval()
    loss = 0
    P = (len(dataset) - 1) * (len(dataset[0][0])) + len(dataset[-1][0])
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            y_pred = mod(x)
            batch_loss = loss_fun(y_pred, y)
            loss += batch_loss.item() * (len(x) / P)
    return loss


# Used to compute the accuracy over the entire data set
def accuracy(dataset: torch.utils.data.DataLoader,
             mod: torchvision.models,
             device: Union[torch.device, str],
             val_check=False, ep=0):
    mod.eval()
    correct_predictions = 0
    total_samples = 0

    if val_check:
        random.seed(ep)
        dataset = random.sample(dataset, int(len(dataset) / 5))

    with torch.no_grad():
        for inputs, labels in dataset:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mod(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy

def closure_2(dataset,
                device: torch.device,
                mod,
                loss_fun,
                test=False):
    mod.eval()
    with torch.no_grad():
        if isinstance(dataset, Dataset):
            if test == False:
                mb_size = 128
                n_it = int(np.ceil((dataset.P/mb_size)))
                L = 0
                for j in range(n_it):
                    dataset.minibatch(j*mb_size,(j+1)*mb_size)
                    x = dataset.x_train_mb.to(device)
                    y = dataset.y_train_mb.to(device)
                    y_pred = mod(x)
                    loss = loss_fun(input=y_pred,target=y).item()
                    L += (len(x)/dataset.P)*loss
            else:
                mb_size = 128
                n_it = int(np.ceil((dataset.P_test / mb_size)))
                L = 0
                for j in range(n_it):
                    dataset.minibatch(j * mb_size, (j + 1) * mb_size,test=True)
                    x = dataset.x_test_mb.to(device)
                    y = dataset.y_test_mb.to(device)
                    y_pred = mod(x)
                    loss = loss_fun(input=y_pred, target=y).item()
                    L += (len(x) / dataset.P) * loss
        else:
            L = 0
            P = (len(dataset)-1) * (len(dataset[0][0])) + len(dataset[-1][0])
            for x, y in dataset:
                x, y = x.to(device), y.to(device)
                y_pred = mod(x)
                loss = loss_fun(y_pred, y)
                L += loss.item() * (len(x) / P)
    return L


def accuracy_2(dataset,
            mod,
            device):
    with torch.no_grad():
        if isinstance(dataset, Dataset):
            mb_size = 128
            correct_predictions = 0
            total_samples = 0
            n_it = int(np.ceil((dataset.P_test / mb_size)))
            for j in range(n_it):
                dataset.minibatch(j * mb_size, (j + 1) * mb_size, test=True)
                inputs = dataset.x_test_mb.to(device)
                labels = dataset.y_test_mb.to(device)
                outputs = mod(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            accuracy = correct_predictions / total_samples
        else:
            P = (len(dataset)-1) * (len(dataset[0][0])) + len(dataset[-1][0])
            tot = 0
            for x, y in dataset:
                x, y = x.to(device), y.to(device)
                y_pred = mod(x)
                _, predicted = torch.max(y_pred, 1)
                tot += (y == predicted).sum().item()
            accuracy = tot / P

    return accuracy




'''
def set_architecture(arch:str, input_dim:int, seed: int):
    torch.manual_seed(seed)
    if arch=='S':
        return nn.Sequential(nn.Linear(input_dim,50),nn.Sigmoid(),nn.Linear(50,1))
    elif arch=='M':
        return nn.Sequential(nn.Linear(input_dim,20),nn.Sigmoid(),nn.Linear(20,20),
                             nn.Sigmoid(), nn.Linear(20,20), nn.Sigmoid(), nn.Linear(20,1))
    elif arch=='L':
        return nn.Sequential(nn.Linear(input_dim,50),nn.Sigmoid(),nn.Linear(50,50),
                             nn.Sigmoid(),nn.Linear(50,50),nn.Sigmoid(),nn.Linear(50,50),
                             nn.Sigmoid(),nn.Linear(50,50), nn.Sigmoid(), nn.Linear(50,1))
    elif arch == 'XL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 50),
                             nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50),
                             nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 1))
    elif arch == 'XXL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 300),
                             nn.Sigmoid(), nn.Linear(300, 300), nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 50),nn.Linear(50, 1))
    elif arch == 'XXXL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 50),nn.Linear(50, 1))

    elif arch == '4XL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 50),nn.Linear(50, 1))

    else:
        raise SystemError('Set an architecture in {S,M,L,XL,XXL,XXXL,4XL} and try again')
'''

'''def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
'''

def get_pretrained_net(arch:str, seed: int, num_classes = 10, pretrained=True):
    torch.manual_seed(seed)
    if arch == 'resnet18':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                                                          progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet18()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        return pretrainedmodel


    elif arch == 'resnet50':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
                                                          progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet50()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        return pretrainedmodel

    elif arch == 'resnet152':
            if pretrained:
                pretrainedmodel = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1,
                                                              progress=True)
                num_ftrs = pretrainedmodel.fc.in_features
                pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
                torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
            else:
                pretrainedmodel = torchvision.models.resnet152()
                num_ftrs = pretrainedmodel.fc.in_features
                pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
                torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
            return pretrainedmodel

    elif arch == 'resnet34':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
                                                          progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet34()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        return pretrainedmodel

    elif arch == 'resnet101':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1,
                                                          progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet101()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        return pretrainedmodel

    elif arch == 'mobilenet_v2':
        pretrainedmodel = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1, progress=True)
        num_ftrs = pretrainedmodel.classifier[1].in_features
        pretrainedmodel.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.classifier[1].weight)
        return pretrainedmodel

    else:
        raise SystemError('Set an architecture resnet or mobilenet and try again')

def set_optimizer(opt: str, model, *args, **kwargs):
    if opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), *args, **kwargs)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), *args, **kwargs)
    elif opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), *args, **kwargs)
    elif opt == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), *args, **kwargs)
    elif opt == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), *args, **kwargs)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), *args, **kwargs)
    elif opt == 'cma':
        optimizer = CMA(model.parameters(), *args, **kwargs)
    elif opt == 'lbfgs':
        optimizer = LBFGS(model.parameters(), *args, **kwargs)
    elif opt == 'ig':
        optimizer = IG(model.parameters(), *args, **kwargs)
    elif opt == 'nmcma':
        optimizer = NMCMA(model.parameters(), *args, **kwargs)
    elif opt=='cmal':
        optimizer = CMA_L(model.parameters(), *args, **kwargs)
    elif opt == 'cmalbd':
        optimizer = CMA_L_BD(model.parameters(), *args, **kwargs)
    else:
        raise SystemError('Optimizer not supported')

    return optimizer


def set_dataset(ds:str, bs:int, RR:bool, seed: int, transform = None, subset = None, to_list = False):
    if transform == None:
        if ds in ['cifar10', 'cifar100']:
            torch.manual_seed(seed)
            transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.RandomRotation(10),
                                                        torchvision.transforms.RandomAffine(0, shear=10,scale=(0.8, 1.2)),
                                                        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2,saturation=0.2),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif ds == 'tinyimagenet':
            transform = transforms.Compose([
                transforms.Resize((64, 64)),  # Tiny ImageNet images are smaller in resolution
                transforms.RandomHorizontalFlip(),  # Data augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    if ds == 'cifar10':
        ds_train = torchvision.datasets.CIFAR10(download=True, root='./', train=True, transform=transform)
        ds_test = torchvision.datasets.CIFAR10(download=True, root='./', train=False, transform=transform)
        if subset is not None:
            inds_train, inds_test = subset
            ds_train = Subset(ds_train,inds_train)
            ds_test = Subset(ds_test,inds_test)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=RR)
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=bs, shuffle=False)
    elif ds == 'cifar100':
        ds_train = torchvision.datasets.CIFAR100(download=True, root='./', train=True, transform=transform)
        ds_test = torchvision.datasets.CIFAR100(download=True, root='./', train=False, transform=transform)
        if subset is not None:
            inds_train, inds_test = subset
            ds_train = Subset(ds_train,inds_train)
            ds_test = Subset(ds_test,inds_test)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=RR)
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=bs, shuffle=False)
    elif ds == 'mnist':
        ds_train = torchvision.datasets.MNIST(download=True, root='./', train=True, transform=transform)
        ds_test = torchvision.datasets.MNIST(download=True, root='./', train=False, transform=transform)
        if subset is not None:
            inds_train, inds_test = subset
            ds_train = Subset(ds_train, inds_train)
            ds_test = Subset(ds_test, inds_test)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=RR)
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=bs, shuffle=False)
    elif ds == 'tinyimagenet':
        train_data = datasets.ImageFolder(root='tiny-imagenet-200/tiny-imagenet-200/train', transform=transform)
        test_data = datasets.ImageFolder(root='tiny-imagenet-200/tiny-imagenet-200/test', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=RR)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=False)

    elif ds == 'stl10':
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize the images to 128x128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Loading the STL-10 dataset
        train_dataset = datasets.STL10(root='./data', split='train', transform=transform, download=True)
        test_dataset = datasets.STL10(root='./data', split='test', transform=transform, download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=RR)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    else:
        raise ValueError('Dataset not supported')
    if to_list == True:
        train_loader = [(x,y) for x,y in train_loader]
        test_loader = [(x,y) for x,y in test_loader]
    return train_loader, test_loader





def get_w(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.numel(param)
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size