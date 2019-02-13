#!/usr/bin/env python3
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from LeNet import LeNet

def train(model, train_loader, optimizer):
    model.train()

    # SGD iteration
    total_loss = 0
    for idx, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = F.nll_loss(output, Y, reduction='sum')
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    return total_loss / len(train_loader.dataset)

def validate(model, valid_loader):

    loss = 0
    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(valid_loader):
            output = model(X)
            loss += F.nll_loss(output, Y, reduction='sum').item()

    return loss / len(valid_loader.dataset)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10, help='train mini-batch size')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='validatio mini-batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    use_cuda = False
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mnist_train = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.batch_size, shuffle=True, **kwargs)

    mnist_test = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.valid_batch_size, shuffle=True, **kwargs)


    model = LeNet()

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    x_data = []
    tr_data = []
    va_data = []


    for epoch in range(args.epochs):
        train_loss=train(model, mnist_train, optimizer)
        valid_loss=validate(model, mnist_test)
        x_data.append(epoch + 1)
        tr_data.append(train_loss)
        va_data.append(valid_loss)
        print('Epoch = ', epoch + 1, '\tTrain Loss = ', train_loss, '\tValid Loss = ', valid_loss)

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(x_data, tr_data,  label='Training error')
    ax.plot(x_data, va_data,  label='Validation error')
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('error', fontsize=10)
    leg = ax.legend();
    fig.savefig('cnn_error.png')   # save the figure to file
    plt.close(fig)    # close the figure
