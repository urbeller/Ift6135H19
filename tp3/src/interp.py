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


def pick_samples(model, device, z_dim, ndx_list):
  torch.manual_seed(1977)
  
  sorted_list = sorted(ndx_list)

  params = []
  d = dict()
  j = 0
  for i in sorted_list:
    while j < i + 1:
      noise = Variable(torch.randn(1, z_dim) ).to(device)
      j = j + 1
    
    params.append(noise)
    d[i] = noise

  return d

def interpolate(model, device, z_dim, z0, z1, steps):
  for s in range(steps+1):
    eps = s / steps
    z_i = z1 * eps + z0 * (1 - eps)

    sample = vae.decode(z_i).cpu()
    save_image(sample.data.view(1, 3, 32, 32), 'results/interp-0' + '{:02}'.format(s) + '.png' )

def interpolate_image(model, device, z_dim, z0, z1, steps):
  x0 = vae.decode(z0).cpu()
  x1 = vae.decode(z1).cpu()

  for s in range(steps+1):
    eps = s / steps
    x_i = x1 * eps + x0 * (1 - eps)

    save_image(x_i.data.view(1, 3, 32, 32), 'results/interp-data-0' + '{:02}'.format(s) + '.png' )

def save_samples(model, device, samples, z_dim):
  for k, v in samples.items():
    x = model.decode(v).cpu()      
    save_image(x.data.view(1, 3, 32, 32), 'results/recons-0' + '{:02}'.format(k) + '.png' )

def interp_disent(model, device, z, z_dim, ll):
  nrows = z_dim
  ncols = 10

  for r in ll:
    for c in range(ncols):
      eps = torch.zeros(1, z_dim)
      eps[0,r-1] = c 
      z_prime = z + eps
      sample = model.decode(z_prime).cpu()
      save_image(sample.data.view(1, 3, 32, 32), 'results/disent-0' + '{:02}'.format(r) + '-' + '{:02}'.format(c) + '.png' )

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_model', type=str, default="", help='Path to a saved model')

  args = parser.parse_args()


  z_dim = 100

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  using_cuda = (device == "cuda")
  print("Using ", device)

  vae = models.VAE(device, z_dim = z_dim)

  vae.load_state_dict(torch.load( args.use_model , map_location='cpu') )
  vae.eval()
  vae.to(device)
  ndices =  [26,14366 , 11916, 342,1372, 14340, 13915, 11916, 11362, 10061, 8724, 6935, 1077, 13193, 11913, 10061, 2727]
  samples = pick_samples(vae, device, z_dim, ndices)

  save_samples(vae, device, samples, z_dim)
  x0 , x1 = (14340, 10061)
  #interpolate(vae, device, z_dim, samples[x0], samples[x1], 10)
  #interpolate_image(vae, device, z_dim, samples[x0], samples[x1], 10)
  ll=[12,50,52, 61, 72, 75, 85]
  interp_disent(vae, device, samples[2727], z_dim, ll)

