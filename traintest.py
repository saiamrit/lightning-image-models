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
from trainer import AlexNetLit
import argparse

import warnings
warnings.filterwarnings("ignore")

pl.seed_everything(1)

wandb.login()

def main(args):
    wandb_logger = WandbLogger(project="lit-alexnet")

    cifar = CIFAR10Data(data_dir=args.data_root, batch_size=args.batch_size)

    cifar.prepare_data()
    cifar.setup()

    # grab samples to log predictions on
    # samples = next(iter(cifar.val_dataloader()))
    lr = args.learning_rate
    model = AlexNetLit(out_dim=10, lr=lr)

    if args.earlystop:
        trainer = pl.Trainer(
        logger=wandb_logger,    # W&B integration
        log_every_n_steps=50,   # set the logging frequency
        gpus=1,                # use all GPUs
        max_epochs=args.epochs,           # number of epochs
        deterministic=True,
        callbacks=[EarlyStopping(monitor="VALIDATION LOSS", mode="min")]#, ImagePredictionLogger(samples)]
        )
    else:
        trainer = pl.Trainer(
        logger=wandb_logger,    # W&B integration
        log_every_n_steps=50,   # set the logging frequency
        gpus=1,                # use all GPUs
        max_epochs=args.epochs,           # number of epochs
        deterministic=True
        )

    # fit the model
    trainer.fit(model, cifar)

    # evaluate the model on a test set
    trainer.test(model, cifar)  # uses last-saved model

    trainer.save_checkpoint("AlexNetCifar.pth")

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for training the model")
    parser.add_argument('--epochs', action='store', type=int, default=10, help="epochs to train the model for")
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help="start learning rate")
    parser.add_argument('--gpus', action='store', type=int, default=1, help="number of gpus to use for training")
    parser.add_argument('--data_root', action='store', type=str, default="./", help="location of the dataset")
    parser.add_argument('--batch_size', action='store', type=int, default=128, help="Training batch size")
    parser.add_argument('--earlystop', action='store_true', default=False, help="whether to use early stopping callback")
    # parser.add_argument('--output_path', action='store', type=str, default="/home2/sdokania/all_projects/occ_artifacts/", help="Model saving and checkpoint paths")
    # parser.add_argument('--exp_name', action='store', type=str, default="initial", help="Name of the experiment. Artifacts will be created with this name")
    # parser.add_argument('--encoder', action='store', type=str, default="efficientnet-b0", help="Name of the Encoder architecture to use")
    # parser.add_argument('--decoder', action='store', type=str, default="decoder-cbn", help="Name of the decoder architecture to use")
    args = parser.parse_args()

    # print(args.cdim, args.hdim, args.pdim, args.data_root, args.batch_size)
    main(args)