#!/usr/bin/env python3

import numpy as np

class NN:
    def __init__(self,in_dim=784, hidden_dim=1024, out_dim=10):
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim


    def initialize_weights(self, init_type='zero'):
        self.b1 = np.zeros((1, self.hidden_dim))
        self.b2 = np.zeros((1, self.output_dim))
    
        if init_type == 'gauss':
            self.w1 = np.random.randn(self.input_dim, self.hidden_dim)
            self.w2 = np.random.randn(self.hidden_dim, self.output_dim)

        elif init_type == 'glorot':
            pass
        else:
            self.w1 = np.zeros((self.input_dim, self.hidden_dim))
            self.w2 = np.zeros((self.hidden_dim, self.output_dim))

    def forward(self,input):
        z1 = input.dot(self.w1) + self.b1
        a1 = self.activation(z1)
        output = a1.dot(self.w2) + self.b2
        out_probs = self.softmax(output)

        return out_probs

    def activation(self, x):
        return np.tanh(x)

    def loss(self,prediction, true_labels):
        data_size = true_labels.size
        logprobs = -np.log(prediction[range(data_size),true_labels])
        the_loss = logprobs.mean()

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


    def train(self, train_set, va_set, lr, batch_size=100):
        train_data = train_set[0]
        train_labels = train_set[1]

        # SGD
        for i in range(0, train_data.shape[0], batch_size):
            print(i, "--", i+batch_size)


    def test(self, test_set):
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default=None, help='path to .npy data')

    args = parser.parse_args()
    tr, va, te = np.load(args.datafile)
    
    mlp = NN(hidden_dim= 1024, in_dim=784)
    mlp.initialize_weights(init_type = 'gauss')
    mlp.train(tr, va, 0.01)