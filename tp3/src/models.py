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


class VAE(nn.Module):
  def __init__(self, device, image_channels=3, h_dim=256, z_dim=100):
    super(VAE, self).__init__()

    self.device = device

    self.encoder = nn.Sequential(
    nn.Conv2d(image_channels, 8, 3, 1, 1),
    nn.BatchNorm2d(8),
    nn.ELU(),
    nn.Conv2d(8, 16, 3, 2, 1),
    nn.BatchNorm2d(16),
    nn.ELU(),

    nn.Conv2d(16, 16, 3, 1, 1),
    nn.BatchNorm2d(16),
    nn.ELU(),
    nn.Conv2d(16, 32, 3, 2, 1),
    nn.BatchNorm2d(32),
    nn.ELU(),

    nn.Conv2d(32, 64, 3, 1, 1),
    nn.BatchNorm2d(64),
    nn.ELU(),
    nn.Conv2d(64, 128, 3, 2, 1),
    nn.BatchNorm2d(128),
    nn.ELU()
    )
    
    self.fc_enc = nn.Sequential(
                nn.Linear(128 * 4 * 4, h_dim),
                nn.BatchNorm2d(h_dim),
                nn.ELU()
            )

    self.fc_mu = nn.Linear(h_dim, z_dim) 
    self.fc_logvar = nn.Linear(h_dim, z_dim) 
    self.fc_dec1 = nn.Linear(z_dim, h_dim) 
    self.fc_dec2 = nn.Linear(h_dim, 128 * 4 * 4) 


    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(128, 64, 4, 2, 1),
      nn.BatchNorm2d(64),
      nn.ELU(),

      nn.ConvTranspose2d(64, 32, 3, 1, 1),
      nn.BatchNorm2d(32),
      nn.ELU(),

      nn.ConvTranspose2d(32, 16, 4, 2, 1),
      nn.BatchNorm2d(16),
      nn.ELU(),

      nn.ConvTranspose2d(16, 16, 3, 1, 1),
      nn.BatchNorm2d(16),
      nn.ELU(),

      nn.ConvTranspose2d(16, 8, 4, 2, 1),
      nn.BatchNorm2d(8),
      nn.ELU(),

      nn.ConvTranspose2d(8, image_channels, 3, 1, 1),
      nn.Sigmoid()
    )

    self.fc_dec = nn.Sequential(
                nn.Linear(z_dim, h_dim),
                nn.BatchNorm2d(h_dim),
                nn.ELU(),
                nn.Linear(h_dim, 128 * 4 * 4),
                nn.BatchNorm2d(128 * 4 * 4),
                nn.ELU()
            )

    self.z_dim = z_dim
    self.h_dim = h_dim

  def encode(self, x):
    conv_out = self.encoder(x)
    h = self.fc_enc(conv_out.view(-1, 128 * 4 * 4))

    return self.fc_mu(h), self.fc_logvar(h)

  def decode(self, latent):
    h = self.fc_dec(latent)
    output = self.decoder(h.view(-1, 128, 4, 4))

    return output

  def sample_latent(self, mu, logvar):
    var = torch.exp(logvar)
    std_z = torch.from_numpy(np.random.normal(0, 1, size=var.size())).type(torch.FloatTensor).to(self.device)
    
    return mu + var * Variable(std_z, requires_grad=False)

  def forward(self, x):
    mu, logvar = self.encode(x)
    latent = self.sample_latent(mu, logvar)
    recons = self.decode(latent)

    return recons, mu, logvar


