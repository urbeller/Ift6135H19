#!/usr/bin/env python3
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from vgg import Vgg

def split_data(data_loader, valid_prop=0.1, bs=64):
    if(valid_prop > 1 or valid_prop < 0): 
        valid_prop = 0
        
    split = int((1 - valid_prop) * len(data_loader.dataset))
    index_list = list(range(len(data_loader.dataset)))
    train_idx, valid_idx = index_list[:split], index_list[split:]
    tr_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    
    train_loader = DataLoader(data_loader.dataset, batch_size=bs, sampler=tr_sampler, drop_last=True)
    valid_loader = DataLoader(data_loader.dataset, batch_size=bs, sampler=val_sampler, drop_last=True)
    
    return (train_loader, valid_loader)


def load_images(batch_size=64, root="../data/cat_dog/trainset"):

    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_set = datasets.ImageFolder(root=root, transform=transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader  


def train(model, train_loader, optimizer, epoch):
    model.train()

    # SGD iteration
    total_loss = 0
    n_data = 0 
    for idx, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = F.nll_loss(output, Y, reduction='sum')
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_data += X.size(0)

        if idx % 30 == 0:
            print("Epoch = ",epoch, "\titeration : ", idx + 1, "/", len(train_loader), "\tLoss = ", loss.item())

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
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    data_loader=load_images(root="../data/cat_dog/trainset")
    test_loader=load_images(root="../data/cat_dog/testset")

    tr_loader, val_loader = split_data(data_loader, valid_prop = 0.1, bs=64)

    model = Vgg(num_classes=2)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train(model, tr_loader, optimizer, epoch + 1)
        valid_loss = validate(model, val_loader)


