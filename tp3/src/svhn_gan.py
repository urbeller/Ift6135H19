#!/usr/bin/env python3

import torch
from torch import nn
import torchvision.datasets
from torch.optim import Adam
from torch.autograd import grad
from torch.autograd import Variable
from torch.utils.data import dataset
from torch.nn import functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms

import numpy as np

import models
import utils


def compute_gp(device, D, x_real, x_fake, batch_size, lambda_f = 10):
  sz1 = x_real.size(1)
  sz2 = x_real.size(2)
  sz3 = x_real.size(3)
  alpha = torch.rand(batch_size, 1)
  alpha = alpha.expand(batch_size, sz1 * sz2 * sz3).view(batch_size, sz1, sz2, sz3)
  alpha = alpha.to(device)

  interpolates = Variable(alpha * x_real + ((1.0 - alpha) * x_fake), requires_grad=True).to(device)

  out_interp = D(interpolates)

  ones = torch.ones(out_interp.size()).to(device)
  gradients = grad(outputs=out_interp, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
  gradients = gradients.view(gradients.size(0), -1)
  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_f
  
  return gradient_penalty

def train(device, D, G, train_loader, epochs=100, g_iters=10000, d_iters = 100):
  latent_dim = 100

  if device == 'cuda':
    latent_loss = nn.BCELoss().cuda()
  else:
    latent_loss = nn.BCELoss()

  D.train()
  G.train()
  
  d_optim = torch.optim.Adam(D.parameters(), lr=0.001)
  g_optim = torch.optim.Adam(G.parameters(), lr=0.001)

  one = torch.tensor(1.0)
  mone = torch.tensor(-1.0)

  if device == 'cuda':
    one = one.cuda()
    mone = mone.cuda()

  for g_ndx in range(g_iters):

    # For each G iteration, compute some D iterations.
    utils.set_req_grad(D, True)
    for d_ndx in range(d_iters):
      D.zero_grad()

      X, _ = next(iter(train_loader))
      batch_size = X.shape[0]
  
      x_real = Variable(X.to(device))
      
      # Train D on real data.
      d_real = D(x_real)
      d_real = d_real.mean()
      d_real.backward(mone)

      # Train D on fake images.
      noise = Variable(torch.randn(batch_size, latent_dim)).to(device)
      x_fake = Variable(G(noise))
      d_fake = D(x_fake)
      d_fake = d_fake.mean()
      d_fake.backward(one)

      # Gradient penalty
      gp_loss = compute_gp(device, D, x_real, x_fake, batch_size)
      gp_loss.backward()

      d_optim.step()



    ##
    ## G train.
    utils.set_req_grad(D, False)

    G.zero_grad()
    noise = Variable(torch.randn(batch_size, latent_dim)).to(device)
    x_noise = Variable(G(noise))
    g_fake = G(x_noise)
    g_fake = g_fake.mean()
    g_fake.backward(mone)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_model', type=str, default="", help='Path to a saved model')
  parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

  args = parser.parse_args()

  z_dim = 100
  epochs = args.epochs

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  using_cuda = (device == "cuda")
  print("Using ", device)

  D = models.Discriminator(device, z_dim = z_dim)
  G = models.Generator(device, z_dim = z_dim)

  train_data, valid_data, test_data = utils.get_data_loader("svhn", 32)
  train(device, D, G, train_data)
