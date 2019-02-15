#!/usr/bin/env python3
import numpy as np
import PIL
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


def load_images(batch_size=64, root="../data/cat_dog/trainset", shuffle=False, **kwargs):

    transform = transforms.Compose([
        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        torchvision.transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_set = datasets.ImageFolder(root=root, transform=transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return data_loader  

def load_test_images(batch_size=64, root="../data/cat_dog/trainset", **kwargs):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_set = datasets.ImageFolder(root=root, transform=transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, **kwargs)

    return data_loader 

def train(device, model, train_loader, optimizer):
    model.train()

    # SGD iteration
    total_loss = 0
    n_data = 0 
    for idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = F.nll_loss(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 

        if idx % 30 == 0:
            print("\t\t\titeration : ", idx + 1, "/", len(train_loader), "\tLoss = ", loss.item())

    return total_loss / len(train_loader.dataset)


def validate(device, model, valid_loader):

    loss = 0
    n_data = 0
    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(valid_loader):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss += F.nll_loss(output, Y).item() 

        return loss / len(valid_loader.dataset)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='train mini-batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA')

    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Loading and spliting data ... ", end='')
    data_loader=load_images(root="../data/cat_dog/trainset", **kwargs)
    test_loader=load_images(root="../data/cat_dog/testset", **kwargs)

    tr_loader, val_loader = split_data(data_loader, valid_prop = 0.2, bs=10)
    print("done")

    print("Creating the model ... ", end='')
    model = Vgg(num_classes=2).to( device )
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    print("done")


    model.load_state_dict(torch.load('model.pt'))
    for epoch in range(args.epochs):
        print("[", device, "] | ", "Epoch = ", epoch + 1)
        train_loss = train(device, model, tr_loader, optimizer)
        valid_loss = validate(device, model, val_loader)
        
        print("\t\t -->", train_loss, valid_loss)

