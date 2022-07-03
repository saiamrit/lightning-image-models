<div align="center">    
 
# Pytorch Lightning Image Classification Models
 
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
 <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.9+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
 <a href="https://pytorch.org/get-started/locally/"><img alt="W & B" src="https://img.shields.io/badge/-WandB-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white"></a>
</div>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb)

This repository contains image classification models implemented in Pytorch Lightning. The logging is done with Weights and Biases, integrated to track the metrics and losses. The aim is also to present a clean templete for Pytorch Lightning projects, and present modular and structured code for projects build with Pytorch lightning.

## Requirements
```
 opencv=4.1.2
 pytorch=1.9.1
 torchvision=0.10.1
 pytorch-lightning=1.5.10
 scikit-image=0.15.0
 wandb
```

## Models

- **AlexNet**
## Setting up the Project

The project was built in a conda environment. So it is recommended but not necessary to have anaconda/miniconda installed. The ```environment.yml``` has the necessary environment packages.

Clone the repository to your local using,
```bash
git clone https://github.com/saiamrit/lightning-image-models.git
cd lightning-image-models
```
Then setup the conda environment using,
```bash
conda env create -f environment.yml
```
## Running

Once the environment is created, activate the environment and run the ```traintest.p```


## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/saiamrit/lightning-image-models.git

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
