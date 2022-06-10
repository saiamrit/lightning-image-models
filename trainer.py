import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms, datasets

import pytorch_lightning as pl
import torchmetrics
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *

pl.seed_everything(1)

class AlexNetLit(pl.LightningModule):
    def __init__(self, out_dim, lr):# data_dir, batch_size):
        super().__init__()
        self.model = AlexNet(out_dim)
        
        self.loss = nn.CrossEntropyLoss()        
        self.save_hyperparameters()
        
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        
    def forward(self, x):
        x = self.model.feature_extractor(x)
        feat = x.view(x.shape[0],-1)
        x = self.model.classifier(feat)

        return x, feat
    
    def configure_optimizers(self):
        print(self.hparams['lr'])
        return torch.optim.Adam(self.parameters(), lr = self.hparams['lr'])
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_pred, feat = self(images)
        preds = torch.argmax(y_pred, 1)
#         print(preds.shape, labels.shape)
        
        loss = self.loss(y_pred, labels)
        acc = self.train_accuracy(y_pred, labels)
        
        self.log('TRAIN LOSS', loss, on_epoch = True)
        self.log('TRAIN ACCURACY', acc, on_epoch = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_pred, feat = self(images)
#         print(y_pred.shape, labels.shape)
        preds = torch.argmax(y_pred, 1)
#         print(preds.shape, labels.shape)
        
        loss = self.loss(y_pred, labels)
        acc = self.valid_accuracy(y_pred, labels)
        
        self.log('VALIDATION LOSS', loss, on_step = False, on_epoch= True)
        self.log('VALIDATION ACCURACY', acc, on_step = False, on_epoch = True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        y_pred, feat = self(images)
#         print(y_pred.shape, labels.shape)
        preds = torch.argmax(y_pred, 1)
#         print(preds.shape, labels.shape)
        
        loss = self.loss(y_pred, labels)
        acc = self.test_accuracy(y_pred, labels)
        
        self.log('TEST LOSS', loss, on_step = False, on_epoch= True)
        self.log('TEST ACCURACY', acc, on_step = False, on_epoch = True)
        
        return loss