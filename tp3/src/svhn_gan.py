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


def generate_image(G, device, latent_dim, n_images, prefix):
  noise = Variable(torch.randn(n_images, latent_dim) , requires_grad=False).to(device)

  samples = G(noise)
  samples = samples.view(-1, 3, 32, 32)
  samples = samples.mul(0.5).add(0.5)
  save_image(samples.data.view(n_images, 3, 32, 32).cpu(), 'results/sample-' + str(prefix) + '.png', nrow= 10 )

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

def train(device, D, G, train_loader, latent_dim=100, epochs=100, g_iters=10000, d_iters = 5):

  D.train()
  G.train()
  
  d_optim = torch.optim.Adam(D.parameters(), lr=0.001)
  g_optim = torch.optim.Adam(G.parameters(), lr=0.001)

  one = torch.tensor(1.0).to(device)
  mone = torch.tensor(-1.0).to(device)

  for g_ndx in range(g_iters):

    # For each G iteration, compute some D iterations.
    utils.set_req_grad(D, True)
    for d_ndx in range(d_iters):
      D.zero_grad()

      X, _ = next(iter(train_loader))
      batch_size = X.shape[0]
  
      x_real = Variable(X.to(device))
      
      d_real = D(x_real)
      noise = Variable(torch.randn(batch_size, latent_dim)).to(device)
      x_fake = Variable(G(noise))
      d_fake = D(x_fake)
      gp_loss = compute_gp(device, D, x_real, x_fake, batch_size)
      d_loss = 0.5 * (torch.mean((d_real - 1)**2) + torch.mean(d_fake**2)) + gp_loss
      d_loss.backward()
      d_optim.step()

      """
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
      d_loss = d_fake - d_real + gp_loss

      d_optim.step()
      """


    ##
    ## G train.
    utils.set_req_grad(D, False)

    G.zero_grad()
    x_noise = Variable(torch.randn(batch_size, latent_dim)).to(device)
    g_out = Variable(G(x_noise) , requires_grad=True)

    g_fake = D(g_out)
    g_loss = 0.5 * torch.mean((g_fake - 1)**2)
    g_loss.backward()

    """
    g_fake = g_fake.mean()
    g_fake.backward(mone)
    """
    g_optim.step()

    if g_ndx % 10 == 0:
      print("Iter ", g_ndx, "D_loss=", d_loss.mean().cpu().data.numpy(), "G_loss=", g_loss.mean().cpu().data.numpy())
      generate_image(G, device, latent_dim, 100, g_ndx)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_model', type=str, default="", help='Path to a saved model')
  parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

  args = parser.parse_args()

  latent_dim = 100
  epochs = args.epochs

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  using_cuda = (device == "cuda")
  print("Using ", device)

  D = models.Discriminator(device, z_dim = latent_dim)
  G = models.Generator(device, z_dim = latent_dim)

  D.to(device)
  G.to(device)

  train_data, valid_data, test_data = utils.get_data_loader("svhn", 64)
  train(device, D, G, train_data, latent_dim=latent_dim)
