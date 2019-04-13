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
  batch_size = p.shape[0]

  # The interpolation
  alpha = torch.rand(batch_size,1)
  alpha = alpha.expand(-1,model.input_dim)
  interp = Variable( alpha * p + (1 - alpha) * q, requires_grad = True)

  # Get the interpolation through the model
  out_interp = model(interp)
  gradients = grad(outputs=out_interp, inputs=interp,
                   grad_outputs=torch.ones(out_interp.size()),
                   retain_graph=True, create_graph=True, only_inputs=True)[0]

  # Mean/Expectation of gradients
  gradients = gradients.view(gradients.size(0),  -1)
  gradient_norm = gradients.norm(2, dim=1)

  return (gradient_norm - 1)**2

def loss_wd(model, p, q,lambda_fact = 10):
  gp = computeGP(model, p, q)
  return -(p.mean() - q.mean() - lambda_fact * gp.mean())



def loss_jsd(model, p, q):
  return -(torch.log(torch.Tensor([2])) + 0.5 * torch.mean(torch.log(p)) + 0.5 * torch.mean(torch.log(1 - q)) )




def loss_gan(model, p, q):
  return -( torch.mean(torch.log(p)) + torch.mean(torch.log(1 - q)) )




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

    modules= [ nn.Linear(input_dim, hidden_size) ,  nn.Tanh() ]
    for i in range(n_hidden - 1):
      modules.append(nn.Linear(hidden_size, hidden_size) )
      modules.append(nn.Tanh())
						
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

  optim = torch.optim.SGD(model.parameters(), lr=0.001)
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
    p_out = model(p_tensor)
    q_out = model(q_tensor)

    loss = loss_func(model, p_out, q_out)
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
  p_out = model(p_tensor)
  q_out = model(q_tensor)

  return loss_fn(model, p_out, q_out)
  
def q_1_3():
  epochs = 100
  batch_size=512
  hidden_size = 32
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

  plt.show()

if __name__ == '__main__':
  #input_dim = next(iter(p)).shape[1]
  q_1_3()
"""
  D = Discriminator(input_dim=input_dim, hidden_size=40, n_hidden=3)
  train(D, p, q, loss_wd, batch_size=batch_size, epochs=1000)

  px = next(iter(p))
  qx = next(iter(q))
  out = loss_wd(D, px, qx)
  print(out)

  plt.figure()
  xx =  next(iter(p))
  print(xx)
  xx_tensor = torch.from_numpy(np.float32(xx.reshape(batch_size, input_dim))) 
  r = D(xx_tensor).detach().numpy()
  plt.figure(figsize=(8,4))
  plt.subplot(1,2,1)
  plt.plot(xx,r)
  plt.title(r'$D(x)$')

  plt.show()

"""


