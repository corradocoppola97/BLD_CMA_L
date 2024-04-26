from warnings import filterwarnings
filterwarnings('ignore')
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from utils_BD import set_optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pl.seed_everything(0)

class ResNet(pl.LightningModule):
    def __init__(self,model_name, n_classes,
                 metric, loss_fun, optimizer,
                 device='cuda',
                 pretrained=True,
                 ablate=None,
                 **opt_params):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=pretrained).to(device)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes).to(device)
        self.metric = metric
        self.loss_fun = loss_fun
        self.opt = set_optimizer(optimizer, self.model, **opt_params)
        if ablate is not None:
            for k in range(len(ablate)):
                for p in eval('self.model.layer'+str(ablate[k])+'.parameters()'):
                    p.requires_grad = False


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x,y = x.to(device), y.to(device)
        logits = self(x)
        loss = self.loss_fun(logits, y)
        acc = self.metric(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        dataloader = self.trainer.train_dataloader
        #To be completed


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)
        acc = self.metric(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return self.opt