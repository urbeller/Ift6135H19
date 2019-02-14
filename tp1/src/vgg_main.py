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

    
    train_loader = DataLoader(data_loader.dataset, batch_size=bs, sampler=tr_sampler)
    valid_loader = DataLoader(data_loader.dataset, batch_size=bs, sampler=val_sampler)
    
    return (train_loader, valid_loader)


def load_images(image_size=32, batch_size=64, root="../data/cat_dog/trainset"):

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_set = datasets.ImageFolder(root=root, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader  


if __name__ == '__main__':

    data_loader=load_images(root="../data/cat_dog/trainset")
    test_loader=load_images(root="../data/cat_dog/testset")

    tr_loader, val_loader = split_data(data_loader, valid_prop = 0.1, bs=64)

    model = Vgg(num_classes=2)
