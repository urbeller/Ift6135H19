#!/usr/bin/env python3

import numpy as np

class NN:
    def __init__(self,in_dim=784, hidden_dim=1024, out_dim=10, init_type='zero'):
        self.init_type = init_type
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim
        self.initialize_weights()

    def initialize_weights(self):
        self.b1 = np.zeros((1, self.input_dim))
        self.b2 = np.zeros((1, self.output_dim))
    
        if self.init_type == 'gauss':
            self.w1 = np.random.randn(self.input_dim, self.hidden_dim)
            self.w2 = np.random.randn(self.hidden_dim, self.output_dim)

        elif self.init_type == 'glorot':
            pass
        else:
            self.w1 = np.zeros((self.input_dim, self.hidden_dim))
            self.w2 = np.zeros((self.hidden_dim, self.output_dim))

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
    
    mlp = NN(hidden_dim= 1024, init_type='gauss')
    print(mlp.w1)