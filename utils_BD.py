import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from torchsummaryX import summary
import csv
import ast
import torchvision
from typing import Union
from cmalight import CMA_L
from cmalight_BD import CMA_L_BD

# Used to compute the loss function over the entire data set
def closure(data_loader: torch.utils.data.DataLoader,
            model: torchvision.models,
            criterion: torch.nn,
            device: Union[torch.device, str]):
    loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            batch_loss = criterion(y_pred, y)
            loss += batch_loss.item() * (len(x) / len(data_loader.dataset))
    return loss


# Used to compute the accuracy over the entire data set
def accuracy(data_loader: torch.utils.data.DataLoader,
             model: torchvision.models,
             device: Union[torch.device, str]):
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy


def set_optimizer(opt: str, model: torchvision.models, *args, **kwargs):
    if opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), *args, **kwargs)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), *args, **kwargs)
    elif opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), *args, **kwargs)
    elif opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), *args, **kwargs)
    elif opt == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), *args, **kwargs)
    elif opt == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), *args, **kwargs)
    elif opt == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), *args, **kwargs)
    elif opt == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), *args, **kwargs)
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), *args, **kwargs)
    elif opt == 'rprop':
        optimizer = torch.optim.Rprop(model.parameters(), *args, **kwargs)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), *args, **kwargs)
    elif opt == 'cmal':
        optimizer = CMA_L(model.parameters(), *args, **kwargs)
    elif opt == 'cmalbd':
        optimizer = CMA_L_BD(model.parameters(), *args, **kwargs)
    else:
        raise SystemError('RICORDATI DI SCEGLIERE L OTTIMIZZATORE!  :)')
    return optimizer


def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Actual device: ", device)
    if 'cuda' in device:
        print("Device info: {}".format(str(torch.cuda.get_device_properties(device)).split("(")[1])[:-1])

    return device


def print_model(model, device, save_model_root, input_shape):
    info = summary(model, torch.ones((1, 3, 32, 32)).to(device))
    info.to_csv(save_model_root + 'model_summary.csv')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_csv_history(path):
    objects = []
    with (open(path + 'history.pkl', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    df = pd.DataFrame(objects)
    df.to_csv(path + 'history.csv', header = False, index = False, sep = " ")


def plot_graph(data, label, title, path):
    epochs = range(0, len(data))
    plt.plot(epochs, data, 'orange', label = label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid('on', color = '#cfcfcf')
    plt.tight_layout()
    plt.savefig(path + title + '.pdf')
    plt.close()


def plot_history(history, path):
    plot_graph(history['train_loss'], 'Train Loss', 'Train_loss', path)
    plot_graph(history['val_acc'], 'Val Acc.', 'Val_acc', path)


def extract_history(history_file):
    with open(history_file) as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            l = row[0]
            break
        l = l[:-9]
        train_loss = ast.literal_eval(l)
        val_accuracy = row[498:498 + 250]
        val_accuracy[-1] = val_accuracy[-1][:4]
        val_accuracy[0] = val_accuracy[0][-5:]
        val_accuracy = [float(c.strip('[ ]')) for c in val_accuracy]
    return train_loss, val_accuracy


dataset = 'cifar10'
nets = ['resnet18', 'resnet34', 'resnet50', 'resnet152', 'swin_t', 'swin_b', 'efficientnet_v2_l']
# algos = ['cmal','sgd','adamw','adagrad','adam','rmsprop','adamax']
algos = ['cmal', 'cmalbd', 'sgd', 'adamw', 'adagrad', 'adam', 'rmsprop', 'adamax']
colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'brown', 'yellow', 'pink', 'black']


def make_plot(dataset, net, epochs):
    plt.figure()
    for j in range(len(algos)):
        stats = torch.load('history_' + algos[j] + '_' + dataset + '_' + net + '_prova.txt')
        if epochs == True:
            plt.plot([_ for _ in range(1, len(stats['train_loss']) + 1)], stats['train_loss'], color = colors[j])
        else:
            plt.plot(stats['elapsed_time_noVAL'], stats['train_loss'], color = colors[j])
    plt.legend(algos)
    if epochs == True:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('Elapsed time (s)')
    plt.ylabel('Training Loss')
    plt.title(net + '  Training Loss')
    if epochs == True:
        plt.savefig('Loss_' + dataset + '_' + net + '_epochs.png')
    else:
        plt.savefig('Loss_' + dataset + '_' + net + '_time.png')

    plt.figure()
    for j in range(len(algos)):
        stats = torch.load('history_' + algos[j] + '_' + dataset + '_' + net + '_prova.txt')
        if epochs == True:
            plt.plot([_ for _ in range(1, len(stats['train_loss']) + 1)], stats['val_acc'], color = colors[j])
        else:
            plt.plot(stats['elapsed_time_noVAL'], stats['val_acc'], color = colors[j])
    plt.legend(algos)
    if epochs == True:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('Elapsed time (s)')
    plt.ylabel('Validation Accuracy')
    plt.title(net + '  Validation Accuracy')
    if epochs == True:
        plt.savefig('Accuracy_' + dataset + '_' + net + '_epochs.png')
    else:
        plt.savefig('Accuracy_' + dataset + '_' + net + '_time.png')

    plt.figure()
    for j in range(len(algos)):
        stats = torch.load('history_' + algos[j] + '_' + dataset + '_' + net + '_prova.txt')
        if epochs == True:
            plt.plot([_ for _ in range(1, len(stats['val_loss']) + 1)], stats['val_loss'], color = colors[j])
        else:
            plt.plot(stats['elapsed_time_noVAL'], stats['val_loss'], color = colors[j])
    plt.legend(algos)
    if epochs == True:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('Elapsed time (s)')
    plt.ylabel('Validation Loss')
    plt.title(net + '  Validation Loss')
    if epochs == True:
        plt.savefig('Validation Loss_' + dataset + '_' + net + '_epochs.png')
    else:
        plt.savefig('Validation Loss_' + dataset + '_' + net + '_time.png')


'''
for net in nets:
    try:
        make_plot(dataset, net, epochs = True)
        make_plot(dataset, net, epochs = False)
    except:
        print('PASS')
    print('DONE')
'''
