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
  #samples = samples.add(0.5)
  save_image(samples.data.view(n_images, 3, 32, 32).cpu(), 'results/sample-' + str(prefix) + '.png', nrow= 10 )

def compute_gp(device, D, x_real, x_fake, batch_size):
  _alpha = torch.rand(batch_size, 1, device=device, requires_grad=True)
  alpha = _alpha.expand(batch_size, x_real.nelement()/int(batch_size)).contiguous().view(x_real.size())

  interpolates = alpha * x_real + (1.0 - alpha) * x_fake

  out_interp = D(interpolates)

  ones = torch.ones(out_interp.size()).to(device)

  """
  gradients = grad(outputs=out_interp, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0].view(batch_size, -1)
  """
  gradients = grad(outputs=out_interp.mean(), inputs=interpolates, create_graph=True, retain_graph=True)[0]

  gradients = gradients.view(gradients.size(0), -1)
  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
  
  return gradient_penalty

def train(device, D, G, train_loader, batch_size=128, latent_dim=100, epochs=10000, g_iters=1, d_iters = 5):

  D.train()
  G.train()
  
  d_optim = torch.optim.Adam(D.parameters(),lr=2e-4, betas=(.5, .999))
  g_optim = torch.optim.Adam(D.parameters(),lr=2e-4, betas=(.5, .999))

  loss_fn = torch.nn.BCELoss()

  one = torch.tensor(1.0).to(device)
  mone = torch.tensor(-1.0).to(device)
  
  for epoch in range(epochs):

    for idx, (X,_) in enumerate(train_loader):
      batch_size = X.shape[0] 
      step_d = epoch * batch_size + idx + 1
      x_real = Variable(X.to(device))
      
      # Optimize D
      noise = Variable(torch.randn(batch_size, latent_dim)).to(device)
      x_fake = Variable(G(noise).to(device))
      d_fake = D(x_fake.detach())
      d_real = D(x_real)
      gp = compute_gp(device, D, x_real, x_fake, batch_size)
      d_loss = d_fake.mean() - d_real.mean() + 10.0 * gp
      D.zero_grad()
      d_loss.backward()
      d_optim.step()


      # Optimize G after 'd_iters' of D
      if step_d % d_iters == 0:
        noise = Variable(torch.randn(batch_size, latent_dim)).to(device)
        x_fake = Variable(G(noise).to(device))
        d_fake = D(x_fake)
        g_loss = -d_fake.mean()
        
        G.zero_grad()
        D.zero_grad()
        g_loss.backward()
        g_optim.step()

    #if (idx + 1) % 99 == 0:
    print("Epoch", epoch, ", Step ", idx, "D_loss=", d_loss.mean().cpu().data.numpy(), "G_loss=", g_loss.mean().cpu().data.numpy())
    generate_image(G, device, latent_dim, 100, step_d )

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

  D = models.Discriminator(device, z_dim = latent_dim).apply(utils.initialize_weights)
  G = models.Generator(device, z_dim = latent_dim).apply(utils.initialize_weights)

  D.to(device)
  G.to(device)

  batch_size = 64
  train_data, valid_data, test_data = utils.get_data_loader("svhn", batch_size)
  train(device, D, G, train_data, batch_size=batch_size, latent_dim=latent_dim)
