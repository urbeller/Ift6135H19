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
        X, Y = Variable(X.to(device)), Variable(Y.to(device))

        optimizer.zero_grad()
        output = model(X)
        loss = F.nll_loss(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_data += X.size(0)
        
        if idx % 400 == 0:
            print("\t\t\titeration : ", idx + 1, "/", len(train_loader), "\tLoss = ", loss.item())

    return total_loss / n_data


def validate(device, model, valid_loader):

    loss = 0
    n_data = 0
    valid_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(valid_loader):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss += F.nll_loss(output, Y).item()  * X.size(0)
            n_data += X.size(0)
            
            max_vals, max_indices = torch.max(output,1)
            valid_acc += (max_indices == Y).cpu().sum().data.numpy()

        return (loss / n_data, valid_acc / n_data)


def test_data(device, model, loader):

  out_list = np.empty(shape=[0,2])
  model.eval()
  with torch.no_grad():
    for idx, (X, Y) in enumerate(loader):
      X, Y = X.to(device), Y.to(device)
      output = model(X)
      out_list = np.append(out_list, output.data.cpu().numpy(), axis=0)

  return np.exp(out_list)

def label_result(pred, test_loader, labels_names):
  labels_int=np.argmax(pred, axis=1)
  labels_str = [labels_names[x] for x in labels_int.tolist()]
  img_ids = [os.path.splitext(os.path.basename(path[0]))[0] for path in test_loader.dataset.imgs]
  result = list(zip(img_ids,labels_str))
  
  result.sort(key=lambda tup: int(tup[0]))
  
  return result

def to_csv(the_list,path):
  with open(path,'w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['id','label'])
    for row in the_list:
      csv_out.writerow(row)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='train mini-batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA')

    args = parser.parse_args()


    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Loading and spliting data ... ", end='')
    data_loader=load_images(root="trainset/", batch_size=10, shuffle = True, **kwargs)
    #test_loader=load_images(root="cat_dog/testset", batch_size=10, shuffle = False, **kwargs)

    tr_loader, val_loader = split_data(data_loader, valid_prop = 0.2, bs=10)
    print("done")

    print("Creating the model ... ", end='')
    model = Vgg(num_classes=2).to( device )
    best_model = Vgg(num_classes=2).to( device )
    optimizer = optim.SGD(model.parameters(), lr=1E-2)
    print("done")


    ## Train
    x_data = []
    tr_data = []
    va_data = []

    best_acc = 0
    for epoch in range(35):
           print("[", device, "] | ", "Epoch = ", epoch + 1)
           train_loss = train(device, model, tr_loader, optimizer)
           (valid_loss, valid_accuracy) = validate(device, model, val_loader)
           
           if valid_accuracy > best_acc:
              best_acc = valid_accuracy
              torch.save(model.state_dict(), 'model.pt')
              best_model.load_state_dict(model.state_dict())
              
           x_data.append(epoch + 1)
           tr_data.append(train_loss)
           va_data.append(valid_loss)
            
           print("\t\t -->", "Train Loss = ", train_loss, "Validation Loss = ", valid_loss, "Accuracy = ", valid_accuracy * 100)


    ## Tests
    test_loader=load_images(root="testset", batch_size=100, shuffle = False, **kwargs)   
    out_l=test_data(device,best_model, test_loader)
    result = label_result(out_l, test_loader, ['Cat','Dog'])
    to_csv(result, 'cat_dog_submission.csv')

    ## Plot error curves.
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(x_data, tr_data,  label='Training error')
    ax.plot(x_data, va_data,  label='Validation error')

    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('error', fontsize=10)
    leg = ax.legend();

    fig.savefig('cat_dog_error.png')   # save the figure to file
    plt.close(fig)    # close the figure
