import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
          
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
          
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
          
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
          
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
          
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
          
            nn.Linear(4096, out_dim)
        )
  
def forward(self, x):
      x = self.feature_extractor(x)
      feat = x.view(x.shape[0],-1)
      x = self.classifier(feat)

      return x, feat