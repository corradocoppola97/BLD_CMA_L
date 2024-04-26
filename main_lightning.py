from warnings import filterwarnings
filterwarnings('ignore')
import torch
from torch import nn
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchmetrics
from DataModule import CIFARDataModule
from NetworkModule import ResNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pl.seed_everything(0)
accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=10).to(device)

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(10),
                                            transforms.RandomAffine(0, shear = 10, scale = (0.8, 1.2)),
                                            transforms.ColorJitter(brightness = 0.2, contrast = 0.2,
                                                                               saturation = 0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (


                                                0.5, 0.5, 0.5))])

data_module = CIFARDataModule(batch_size=128,transform=transform,RR=True,datase_name='cifar10')
data_module.prepare_data()
model = ResNet(model_name='resnet18',n_classes=10,metric=accuracy,loss_fun=nn.functional.cross_entropy,optimizer='adam',lr=1e-3)
#trainer = Trainer(max_epochs=5)
#trainer.fit(model, datamodule=data_module)
