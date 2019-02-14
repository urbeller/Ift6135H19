#!/usr/bin/env python3
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader



import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from LeNet import LeNet

def split_data(data_loader, valid_prop=0.1, bs=64):
    if(valid_prop > 1 or valid_prop < 0): 
        valid_prop = 0
        
    split = int((1 - valid_prop) * len(data_loader.dataset))
    index_list = list(range(len(data_loader.dataset)))
    train_idx, valid_idx = index_list[:split], index_list[split:]
    tr_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    
    train_loader = DataLoader(data_loader.dataset, batch_size=bs, sampler=tr_sampler)
    valid_loader = DataLoader(data_loader.dataset, batch_size=bs, sampler=val_sampler)
    
    return (train_loader, valid_loader)

def train(model, train_loader, optimizer):
    model.train()

    # SGD iteration
    total_loss = 0
    n_data = 0
    for idx, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = F.nll_loss(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_data += X.size(0)
        
    return total_loss / n_data

def validate(model, valid_loader):

    loss = 0
    n_data = 0
    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(valid_loader):
            output = model(X)
            loss += F.nll_loss(output, Y).item() * X.size(0)
            n_data += X.size(0)

        return loss / n_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='train mini-batch size')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='validatio mini-batch size')
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


    train_data, valid_data = split_data(mnist_train, valid_prop=0.2, bs=args.batch_size)

    model = LeNet()

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    x_data = []
    tr_data = []
    va_data = []


    for epoch in range(args.epochs):
        train_loss=train(model, train_data, optimizer)
        valid_loss=validate(model, valid_data)
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
