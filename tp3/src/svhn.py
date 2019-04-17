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

import models
import utils

def train(device, model, train_loader, epochs=100):
  
  if device == 'cuda':
    latent_loss = nn.BCELoss().cuda()
  else:
    latent_loss = nn.BCELoss()

  utils.initialize_weights(model)
  model.train()
  
  optim = torch.optim.Adam(model.parameters(), lr=0.001)

  best_loss = 999999
  for epoch in range(epochs):
    
    train_loss = 0

    for idx, (X,Y) in enumerate(train_loader):
      X = Variable(X.to(device))

      batch_size = X.shape[0]
      optim.zero_grad()
      recons, mu, logvar = model(X)

      # Compute loss
      scaling_fact = X.shape[0] * X.shape[1] * X.shape[2] * X.shape[3]

      """
      bce = latent_loss(recons, X)
      kl = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
      kl /= scaling_fact
      loss = bce +  kl
      """
      #bce = latent_loss(recons.view(batch_size, -1), X.view(batch_size, -1))
      bce = latent_loss(recons, X).mean()
      kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      loss = bce + kl

      loss.backward()
      train_loss += loss.item()
      optim.step()

    print("Epoch: ", epoch, "Loss=", train_loss)
    if train_loss < best_loss:
      best_loss = train_loss
      torch.save(vae.state_dict(), 'vae_model.pth')

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

  vae = models.VAE(device, z_dim = z_dim)

  if args.use_model == "" :
    train_data, valid_data, test_data = utils.get_data_loader("svhn", 32)
    vae.to(device)
    train(device, vae, train_data, epochs=epochs)
    torch.save(vae.state_dict(), 'vae_model_final.pth')

  else:
    sqrt_n_samples = 25
    n_samples = sqrt_n_samples * sqrt_n_samples
    vae.load_state_dict(torch.load( args.use_model , map_location='cpu') )
    vae.eval()
    vae.to(device)

    # Get some samples
    sample = Variable(torch.randn(n_samples, z_dim) )
    sample.to(device)

    sample = vae.decode(sample).cpu()
    print(sample.shape)
    print(sample.max())
    save_image(sample.data.view(n_samples, 3, 32, 32), 'results/sample.png', nrow= sqrt_n_samples )
