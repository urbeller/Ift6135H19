#!/usr/bin/env python3


from __future__ import print_function
import numpy as np
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import matplotlib.pyplot as plt

import samplers



"""
Loss functions used for the questions.
"""
def computeGP(model, p, q):
  batch_size = p.size()[0]

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

def compute_wd(model, p, q, lambda_fact = 10):
  p_out = model(p)
  q_out = model(q)
  gp = computeGP(model, p_out, q_out)
  return q_out.mean() - p_out.mean() + lambda_fact * gp.mean()



def compute_jsd(model, p, q):
  p_out = model(p)
  q_out = model(q)
  return -(torch.log(torch.Tensor([2])) + 0.5 * torch.mean(torch.log(p_out)) + 0.5 * torch.mean(torch.log(1 - q_out)) )




def compute_gan(model, p, q):
  p_out = model(p)
  q_out = model(q)
  return -( torch.mean(torch.log(p_out)) + torch.mean(torch.log(1 - q_out)) )



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



def train(model, p, q, loss_func, batch_size=512, epochs=1000):
  model.train()

  optim = torch.optim.SGD(model.parameters(), lr=0.001)
  dist_p = iter(p)
  dist_q = iter(q)

  for e in range(epochs):
    # p data.
    px = next(dist_p)
    px_tensor = Variable( torch.from_numpy(np.float32(px.reshape(batch_size, model.input_dim))) )

    # q data.
    qx = next(dist_q)
    qx_tensor = Variable( torch.from_numpy(np.float32(qx.reshape(batch_size, model.input_dim))) )

    optim.zero_grad()
    loss = loss_func(model, px_tensor, qx_tensor)
    loss.backward()
    optim.step()


    if e % 100 == 0:
      print("Epoch ", e, "Loss = ", loss.data.numpy())



 
if __name__ == '__main__':
  batch_size=512
  p = samplers.distribution1(0, batch_size)
  q = samplers.distribution1(1, batch_size)
  input_dim = next(iter(p)).shape[1]

  jsd_model = Discriminator(input_dim=input_dim, hidden_size=40, n_hidden=3)
  train(jsd_model, p, q, compute_gan, batch_size=batch_size, epochs=1000)


  plt.figure()
  xx =  next(iter(p))
  print(xx)
  xx_tensor = torch.from_numpy(np.float32(xx.reshape(batch_size, input_dim))) 
  r = jsd_model(xx_tensor).detach().numpy()
  plt.figure(figsize=(8,4))
  plt.subplot(1,2,1)
  plt.plot(xx,r)
  plt.title(r'$D(x)$')

  plt.show()




