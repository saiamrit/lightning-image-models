import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

import pytorch_lightning as pl
import torchmetrics
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import *

pl.seed_everything(1)

wandb.login()

def main():
	wandb_logger = WandbLogger(project="lit-alexnet")

	cifar = CIFAR10Data(data_dir='./', batch_size=128)

	cifar.prepare_data()
	cifar.setup()

	# grab samples to log predictions on
	# samples = next(iter(cifar.val_dataloader()))

	model = AlexNetLit(out_dim=10, lr=0.001)

	trainer = pl.Trainer(
	    logger=wandb_logger,    # W&B integration
	    log_every_n_steps=50,   # set the logging frequency
	    gpus=1,                # use all GPUs
	    max_epochs=100,           # number of epochs
	    deterministic=True,
	    callbacks=[EarlyStopping(monitor="VALIDATION LOSS", mode="min")]#, ImagePredictionLogger(samples)]
	    )

	# fit the model
	trainer.fit(model, cifar)

	# evaluate the model on a test set
	trainer.test(model, cifar)  # uses last-saved model

	trainer.save_checkpoint("AlexNetCifar.pth")

	wandb.finish()

if __name__ == '__main__':
	main()