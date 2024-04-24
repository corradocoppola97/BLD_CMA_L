from warnings import filterwarnings
filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pl.seed_everything(0)

class CIFARDataModule(LightningDataModule):
    def __init__(self, datase_name, batch_size, transform, RR):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.RR = RR
        self.datase_name = datase_name

    def prepare_data(self):
        if self.datase_name == 'cifar10':
            self.train = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            self.val = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        elif self.datase_name == 'cifar100':
            self.train = datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform)
            self.val = datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform)
        else:
            raise ValueError("Select a dataset in {cifar10, cifar100}")


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=self.RR)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)