#!/usr/bin/env python3

import numpy as np

class NN:
    def __init__(self,in_dim=784, hidden_dim=1024, out_dim=10, init_type='zero'):
        self.init_type = init_type
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        self.a1 = None # caches a1 for backprop.
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

    def forward(self,input):
        z1 = input.dot(self.w1) + self.b1
        self.a1 = self.activation(z1)
        output = self.a1.dot(self.w2) + b2
        out_probs = self.softmax(output)

        return out_probs

    def activation(self,input): lambda x: np.tanh(x)

    def loss(self,prediction, true_labels):
        pass

    def softmax(self,input):
        num = np.exp(input)
        prob = num / np.sum(num, axis=0, keepdims=True)
        
        return prob


    def backward(self,cache,labels):
        pass

    def update(self,grads):
        w1 += -self.lr * grads['w1']
        b1 += -self.lr * grads['b1']
        w2 += -self.lr * grads['w2']
        b2 += -self.lr * grads['b2']


    def train(self, train_set, va_set, lr):
        pass

    def test(self, test_set):
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