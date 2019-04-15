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
                nn.Dropout(p=0.1),
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
                nn.Dropout(p=0.1),
                nn.ELU(),
                nn.Linear(h_dim, 128 * 4 * 4),
                nn.Dropout(p=0.1),
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



def train(device, model, train_loader, epochs=100):
  
  if device == 'cuda':
    latent_loss = nn.BCELoss().cuda()
  else:
    latent_loss = nn.BCELoss()

  initialize_weights(model)
  model.train()
  
  optim = torch.optim.Adam(model.parameters(), lr=0.001)

  best_loss = 999999
  for epoch in range(epochs):
    
    train_loss = 0

    for idx, (X,Y) in enumerate(train_loader):
      X = Variable(X.to(device))

      optim.zero_grad()
      recons, mu, logvar = model(X)

      # Compute loss
      scaling_fact = X.shape[0] * X.shape[1] * X.shape[2] * X.shape[3]

      bce = latent_loss(recons, X)
      variance = torch.exp(logvar)
      variance_sq = variance * variance
      kl = 0.5 * torch.mean(mu * mu + variance_sq - torch.log(variance_sq) - 1)
      #kl = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
      #kl /= scaling_fact

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

  vae = VAE(device, z_dim = z_dim)

  if args.use_model == "" :
    train_data, valid_data, test_data = get_data_loader("svhn", 32)
    vae.to(device)
    train(device, vae, train_data, epochs=epochs)
    torch.save(vae.state_dict(), 'vae_model_final.pth')

  else:
    sqrt_n_samples = 35
    n_samples = sqrt_n_samples * sqrt_n_samples
    vae.load_state_dict(torch.load( args.use_model , map_location='cpu') )
    vae.eval()
    vae.to(device)

    # Get some samples
    sample = Variable(torch.randn(n_samples, z_dim))
    sample.to(device)

    sample = vae.decode(sample).cpu()
    print(sample.shape)
    save_image(sample.data.view(n_samples, 3, 32, 32), 'results/sample.png', nrow= sqrt_n_samples )
