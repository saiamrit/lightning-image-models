import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models

from .alexnet import AlexNet

models = {
	"alexnet" : AlexNet
}

def build_model(model_name = 'alexnet'):
	return models[model_name]