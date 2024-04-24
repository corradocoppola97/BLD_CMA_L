import torch
import torchvision


def get_pretrained_net(type_model, num_classes=10):
    if type_model == 'resnet18':
        pretrainedmodel = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                                                      progress = True)
        num_ftrs = pretrainedmodel.fc.in_features
        pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)

    elif type_model == 'resnet50':
        pretrainedmodel = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
                                                      progress = True)
        num_ftrs = pretrainedmodel.fc.in_features
        pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)

    elif type_model == 'resnet152':
        pretrainedmodel = torchvision.models.resnet152(weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1,
                                                       progress = True)
        num_ftrs = pretrainedmodel.fc.in_features
        pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)

    elif type_model == 'mobilenet_v2':
        pretrainedmodel = torchvision.models.mobilenet_v2(
            weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1, progress = True)
        num_ftrs = pretrainedmodel.classifier[1].in_features
        pretrainedmodel.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.classifier[1].weight)

    elif type_model == 'wide_resnet50':
        pretrainedmodel = torchvision.models.wide_resnet50_2(
            weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress = True)
        num_ftrs = pretrainedmodel.fc.in_features
        pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)

    elif type_model == 'efficientnet_v2_l':
        pretrainedmodel = torchvision.models.efficientnet_v2_l(
            weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1, progress = True)
        num_ftrs = pretrainedmodel.classifier[1].in_features
        pretrainedmodel.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.classifier[1].weight)

    elif type_model == 'swin_t':
        pretrainedmodel = torchvision.models.swin_t(weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1,
                                                    progress = True)
        num_ftrs = pretrainedmodel.head.in_features
        pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.head.weight)

    elif type_model == 'swin_v2_t':
        pretrainedmodel = torchvision.models.swin_v2_t(weights = torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1,
                                                       progress = True)
        num_ftrs = pretrainedmodel.head.in_features
        pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.head.weight)

    elif type_model == 'swin_b':
        pretrainedmodel = torchvision.models.swin_b(weights = torchvision.models.Swin_B_Weights.IMAGENET1K_V1,
                                                    progress = True)
        num_ftrs = pretrainedmodel.head.in_features
        pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.head.weight)

    elif type_model == 'swin_v2_b':
        pretrainedmodel = torchvision.models.swin_v2_b(weights = torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1,
                                                       progress = True)
        num_ftrs = pretrainedmodel.head.in_features
        pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.head.weight)

    elif type_model == 'maxvit_t':  # NO X CIFAR10
        pretrainedmodel = torchvision.models.maxvit_t(weights = torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1,
                                                      progress = True)
        num_ftrs = pretrainedmodel.classifier[5].in_features
        pretrainedmodel.classifier[5] = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.classifier[5].weight)

    return pretrainedmodel


import torch.nn as nn


def get_ilaria_net(type_model, num_classes=10):
    if type_model == 'netprova1':
        model = nn.Sequential(nn.Conv2d(3, 48, 3, 1, 1),  # 3x3x3x48 = 1296
                              nn.BatchNorm2d(48),  # 48,48,48 (3 tensori di parm diversi)
                              nn.ReLU(),
                              nn.Conv2d(48, 128, 3, 2, 1),  # 3x3x48x128 = 55296
                              nn.Dropout(0.3),
                              nn.BatchNorm2d(128),  # 128,128,128
                              nn.ReLU(),
                              nn.MaxPool2d(2, 2),
                              nn.Conv2d(128, 96, 3, 1, 1),  # 3x3x128x96 = 110592
                              nn.Dropout(0.2),
                              nn.BatchNorm2d(96),  # 96,96,96
                              nn.ReLU(),
                              nn.Flatten(),
                              nn.Linear(6144, 1024),  # FC 6144x1024 = 6291456
                              nn.Linear(1024, num_classes)  # 1024x10 + 1024(bias) + 10 (3 tensori diversi)
                              )

    if type_model == 'netprova2':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(32 * 32 * 3, 10), #torch.Size([10, 3072]) ->omega, #torch.Size([10]) ->bias
                              nn.Softmax(dim = 1))

    if type_model == 'netprova3':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(32 * 32 * 3, 5), #torch.Size([10, 3072]) ->omega, #torch.Size([10]) ->bias
                              nn.Linear(5, 10),
                              nn.Softmax(dim = 1))
    return model
