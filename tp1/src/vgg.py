import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Vgg(nn.Module):
    def __init__(self, num_classes):
        super(Vgg, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=(3,3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)


        for l in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, 
                  self.conv6,  self.conv7, self.conv8,  self.conv9,  self.conv10,
                  self.conv11, self.conv12, self.conv13, self.fc1, self.fc2, self.fc3]:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)
    
    def forward(self, x):
        x = F.relu( self.conv2( self.conv1(x) ) )
        x = F.max_pool2d(x, 2, 2)

        x = F.relu( self.conv4( self.conv3(x) ) )
        x = F.max_pool2d(x, 2, 2)

        x = F.relu( self.conv7( self.conv6( self.conv5(x) ) ) )
        x = F.max_pool2d(x, 2, 2)
        #x = F.relu( self.conv10( self.conv9( self.conv8(x) ) ) )
        #x = F.max_pool2d(x, 2, 2)
        #x = F.relu( self.conv13( self.conv12( self.conv11(x) ) ) )
        #x = F.max_pool2d(x, 2, 2)

        # Equivalent to 1x1 convolution.
        x = x.view(-1, 128 * 8 * 8)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
