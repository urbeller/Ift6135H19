#!/usr/bin/env python3

import numpy as np

class NN:
    def __init__(self,hidden_dims=(1024,2048), n_hidden=2):
        self.n_hidden = n_hidden

    def initialize_weights(self,n_hidden,dims):
        pass

    def forward(self,input,labels):
        pass

    def activation(self,input):
        z1 = input.dot(self.w1) + self.b1
        a1 = np.tanh(z1)
        output = a1.dot(self.w2) + b2

        return output

    def loss(self,prediction):
        pass

    def softmax(self,input):
        num = np.exp(input)
        prob = num / np.sum(num, axis=0, keepdims=True)
        
        return prob


    def backward(self,cache,labels):
        pass

    def update(self,grads):
        pass

    def train(self):
        pass

    def test(self):
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default=None, help='path to .npy data')

    args = parser.parse_args()
    _tr, _va, _te = np.load(args.datafile)
    tr=_tr[0]; va=_va[0]; te=_te[0]
    
    mlp = NN()
    print(mlp.softmax([1,4,8]))