#!/usr/bin/env python3

import torch
from torch import nn
import torchvision.datasets
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image

import numpy as np

image_transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

def get_data_loader(dataset_location, batch_size):
  trainvalid = torchvision.datasets.SVHN(
  dataset_location, split='train',
  download=True, transform=image_transform)

  trainset_size = int(len(trainvalid) * 0.9)
  trainset, validset = dataset.random_split(
  trainvalid,
  [trainset_size, len(trainvalid) - trainset_size])

  trainloader = torch.utils.data.DataLoader(
  trainset,
  batch_size=batch_size,
  shuffle=True, num_workers=2)

  validloader = torch.utils.data.DataLoader(
  validset,
  batch_size=batch_size,)

  testloader = torch.utils.data.DataLoader(
  torchvision.datasets.SVHN(
  dataset_location, split='test',
  download=True,
  transform=image_transform),
  batch_size=batch_size,)

  return trainloader, validloader, testloader


def initialize_weights(net):
  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      m.weight.data.normal_(0, 0.02)
      m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
      m.weight.data.normal_(0, 0.02)
      m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.02)
      m.bias.data.zero_()

def set_req_grad(model, state):
  for p in model.parameters(): 
    p.requires_grad = state 
