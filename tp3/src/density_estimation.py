#!/usr/bin/env python3


from __future__ import print_function
import numpy as np
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import matplotlib.pyplot as plt
from scipy.spatial import distance

import samplers



"""
Loss functions used for the questions.
"""
def computeGP(model, p, q):
  size = p.shape[0]
  a = torch.empty(size,1).uniform_(0,1)
  z = a * p + (1 - a) * q
  z.requires_grad = True
  out = model(z)

  gradients = grad(outputs=out, inputs=z, grad_outputs=torch.ones( out.size()), create_graph=True, only_inputs=True, retain_graph=True)[0]
  gradients = gradients.view(gradients.size(0), -1)
  gradients_norm = gradients.norm(2, dim=1)

  return gradients_norm

def loss_wd(model, p, q,lambda_fact = 10):
  model.train()
  p_out = model(p)
  q_out = model(q)
  norm_vec = computeGP(model, p, q)
  return -(torch.mean(p_out) - torch.mean(q_out) - (lambda_fact * torch.mean(((norm_vec - 1)**2))) )



def loss_jsd(model, p, q, lambda_fact=0): #lambda_fact is dummy. used for signature compatibility
  p_out = model(p)
  q_out = model(q)
  return -(torch.log(torch.Tensor([[2]])) + 0.5 * torch.mean(torch.log(p_out)) + 0.5 * torch.mean(torch.log(1 - q_out)) )




def loss_gan(model, p, q, lambda_fact=0):
  p_out = model(p)
  q_out = model(q)
  return -( torch.mean(torch.log(q_out)) + torch.mean(torch.log(1 - p_out)) )




"""
Analytical version of the estimated functions.
"""

def jsd(p, q):
  r = p + q
  d1 = p * np.log( 2 * p / r )
  d2 = q * np.log( 2 * q / r )
  d1[np.isnan(d1)] = 0
  d2[np.isnan(d2)] = 0

  return 0.5 * np.sum(d1 + d2)

class Discriminator(nn.Module):
  def __init__(self, input_dim=10, hidden_size=20, n_hidden=3):
    super(Discriminator, self).__init__()

    modules= [ nn.Linear(input_dim, hidden_size) ,  nn.ReLU() ]
    for i in range(n_hidden - 1):
      modules.append(nn.Linear(hidden_size, hidden_size) )
      modules.append(nn.ReLU())
						
    modules.append(nn.Linear(hidden_size, 1) )
    modules.append(nn.Sigmoid())

    self.net = nn.Sequential(*modules)
    self.net.apply(self.init_weights)

    self.input_dim = input_dim

  def init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        m.bias.data.fill_(0.0)

  def forward(self, input):
    return self.net(input)



def train(model, p, q, loss_func, batch_size=512, epochs=1000, log=False):
  model.train()

  #optim = torch.optim.SGD(model.parameters(), lr=0.001)
  optim = torch.optim.Adam(model.parameters(), lr=0.001)
  dist_p = iter(p)
  dist_q = iter(q)

  for e in range(epochs):
    optim.zero_grad()

    # p data.
    px = next(dist_p)

    # q data.
    qx = next(dist_q)

    p_tensor = Variable( torch.from_numpy(np.float32(px.reshape(batch_size, model.input_dim))) )
    q_tensor = Variable( torch.from_numpy(np.float32(qx.reshape(batch_size, model.input_dim))) )

    loss = loss_func(model, p_tensor, q_tensor)
    loss.backward()
    optim.step()


    if log:
      if e % 100 == 0 or (e < 100 and e % 10 == 0):
        print("\tEpoch ", e, "Loss = ", loss.data.numpy())


def test_net(model, loss_fn, p, q, batch_size):
  px = next(iter(p))
  qx = next(iter(q))
  p_tensor = Variable( torch.from_numpy(np.float32(px.reshape(batch_size, model.input_dim))) )
  q_tensor = Variable( torch.from_numpy(np.float32(qx.reshape(batch_size, model.input_dim))) )

  return loss_fn(model, p_tensor, q_tensor, lambda_fact=0)
  
def q_1_3():
  epochs = 100
  batch_size= 512
  hidden_size = 50
  n_hidden = 3
  input_dim = 2
  theta_list = np.linspace(-1, 1, 21, endpoint=True)

  loss_fn = loss_jsd
  outputs = []
  for theta in theta_list:
    print("Theta = ", theta)
    p = samplers.distribution1(0, batch_size)
    q = samplers.distribution1(theta, batch_size)

    # Train
    D = Discriminator(input_dim=input_dim, hidden_size=hidden_size, n_hidden=n_hidden)
    train(D, p, q, loss_fn, batch_size=batch_size, epochs=epochs)

    # Test
    out = test_net(D, loss_fn, p, q, batch_size)
   
    # Because we minimized the -loss, we must reinvert it here.
    outputs.append( -out.item() )

  plt.figure()

  plt.figure(figsize=(8,4))
  plt.subplot(1,2,1)
  plt.plot(theta_list,outputs)
  plt.title(r'$D(x)$')

  #plt.show()
  plt.savefig('plot.png') 
  plt.close()


def q_1_4():
  epochs = 1000
  batch_size=512
  hidden_size = 50
  n_hidden = 3
  input_dim = 1

  loss_fn = loss_gan
  f0 = samplers.distributionGaussian(batch_size)
  f1 = samplers.distribution4(batch_size)

  # Train
  D = Discriminator(input_dim=input_dim, hidden_size=hidden_size, n_hidden=n_hidden)
  train(D, f0, f1, loss_fn, batch_size=batch_size, epochs=epochs, log=True)

  # Test
  f = lambda x: torch.tanh(x*2+1) + x*0.75
  d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
  N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)

  batch_size=1000
  
  xx = np.linspace(-5,5,batch_size)
  f0_x_tensor = Variable( torch.from_numpy(np.float32(xx.reshape(batch_size, input_dim))) )
  D_x = D(f0_x_tensor)
  f1_est = N(f0_x_tensor) * D_x / (1 - D_x)


  # Plot the discriminator output.
  r = D_x.detach().numpy() 
  plt.figure(figsize=(8,4))
  plt.subplot(1,2,1)
  plt.plot(xx,r)
  plt.title(r'$D(x)$')

  estimate = f1_est.detach().numpy() 

  # Plot the density.
  plt.subplot(1,2,2)
  plt.plot(xx,estimate)
  plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
  plt.legend(['Estimated','True'])
  plt.title('Estimated vs True')
  plt.savefig('plot.png') 
  plt.close()

if __name__ == '__main__':
  #q_1_3()
  q_1_4()


