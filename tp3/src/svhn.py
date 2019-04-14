#!/usr/bin/env python3

import torch
from torch import nn
import torchvision.datasets
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import dataset
import torchvision.transforms as transforms

import numpy as np

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
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


class VAE(nn.Module):
  def __init__(self, image_channels=3, h_dim=256, z_dim=32):
    super(VAE, self).__init__()

    self.encoder = nn.Sequential(
    nn.Conv2d(image_channels, 8, 3, 1, 1),
    nn.ELU(),
    nn.Dropout2d(p=0.1),
    nn.Conv2d(8, 16, 3, 2, 1),
    nn.ELU(),
    nn.Dropout2d(p=0.1),

    nn.Conv2d(16, 16, 3, 1, 1),
    nn.ELU(),
    nn.Dropout2d(p=0.1),
    nn.Conv2d(16, 32, 3, 2, 1),
    nn.ELU(),
    nn.Dropout2d(p=0.1),

    nn.Conv2d(32, 64, 3, 1, 1),
    nn.ELU(),
    nn.Dropout2d(p=0.1),
    nn.Conv2d(64, 128, 3, 2, 1),
    nn.ELU(),
    )
    


    self.fc_enc = nn.Linear(128, h_dim) 
    self.fc_mu = nn.Linear(h_dim, z_dim) 
    self.fc_logvar = nn.Linear(h_dim, z_dim) 
    self.fc_dec1 = nn.Linear(z_dim, h_dim) 
    self.fc_dec2 = nn.Linear(h_dim, 128) 


    self.decoder = nn.Sequential(
      nn.ELU(),
      nn.ConvTranspose2d(128, 64, 4, 2, 1),
      nn.Dropout2d(p=0.1),
      nn.ELU(),

      nn.ConvTranspose2d(64, 32, 3, 1, 1),
      nn.Dropout2d(p=0.1),
      nn.ELU(),

      nn.ConvTranspose2d(32, 16, 4, 2, 1),
      nn.Dropout2d(p=0.1),
      nn.ELU(),

      nn.ConvTranspose2d(16, 16, 3, 1, 1),
      nn.Dropout2d(p=0.1),
      nn.ELU(),

      nn.ConvTranspose2d(16, 8, 4, 2, 1),
      nn.Dropout2d(p=0.1),
      nn.ELU(),

      nn.ConvTranspose2d(8, image_channels, 3, 1, 1),
    )

    self.z_dim = z_dim
    self.h_dim = h_dim

  def encode(self, x):
    conv_out = self.encoder(x)
    h = self.fc_enc(conv_out.view(-1, 128))

    return self.fc_mu(h), self.fc_logvar(h)

  def decode(self, latent):
    h = self.fc_dec1(latent.view(-1, self.z_dim))
    h = self.fc_dec2(h).view(-1, 128, 4, 4)
    output = self.decoder(h)

    return output

  def sample_latent(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    epsilon = torch.empty_like(std).normal_()

    return epsilon.mul(std).add_(mu)

  def forward(self, x):
    mu, logvar = self.encode(x)
    latent = self.sample_latent(mu, logvar)

    return self.decode(latent)



def train(device, model, train_loader):
  model.train()

  optim = torch.optim.Adam(model.parameters(), lr=0.001)

  for idx, (X,Y) in enumerate(train_loader):
    X = Variable(X.to(device))
    img = model(X)


if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  train_data, valid_data, test_data = get_data_loader("svhn", 32)

  vae = VAE()
  train(device, vae, train_data)
