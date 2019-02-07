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
        pass

    def loss(self,prediction):
        pass

    def softmax(self,input):
        pass

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
    print(tr.shape, va.shape, te.shape)