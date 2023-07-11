import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib_resources

class Artefact_detector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=9,stride=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5,stride=1),
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,stride=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5,stride=3),
            nn.Flatten(),
            nn.Linear(in_features=576,out_features=1), # in_feats=576 by following the output size calculations in Pytorch, starting at 224 input
            nn.Sigmoid()
        )

    def training_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self.model(x).squeeze(dim=1)
        loss = F.binary_cross_entropy(y_hat,y.type(torch.float))
        self.log('train_loss',loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self.model(x).squeeze(dim=1)
        loss = F.binary_cross_entropy(y_hat,y.type(torch.float))
        self.log('valid_loss',loss)

    def test_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self.model(x).squeeze(dim=1)
        test_loss = F.binary_cross_entropy(y_hat,y.type(torch.float))
        self.log('test_loss',test_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=1e-4)
    
    def load_default_weights(self):
        pkg = importlib_resources.files("artefact_detector")
        with importlib_resources.as_file(pkg/'default_model.ckpt') as path:
            weights = path
        return weights
