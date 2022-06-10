import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

import pytorch_lightning as pl

class CIFAR10Data(pl.LightningDataModule):
    
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
    def prepare_data(self):
        # download data, train then test
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            cifar = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar, [45000, 5000])
        if stage == 'test' or stage is None:
            self.cifar_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_val, batch_size=10 * self.batch_size)
        return cifar_val

    def test_dataloader(self):
        cifar_test = DataLoader(self.cifar_test, batch_size=10 * self.batch_size)
        return cifar_test