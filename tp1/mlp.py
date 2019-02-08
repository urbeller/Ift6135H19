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

    def num_params(self):
        n  = 1
        for a in (self.w1, self.w2, self.b1, self.b2):
            n += a.shape[0] * a.shape[1]
        return n

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

    def loss(self,X, true_labels):
        data_size = X.shape[0]
        pred = self.forward(X)
        logprobs = -np.log(pred[range(data_size),true_labels])

        return logprobs.mean()

    def softmax(self,input):
        num = np.exp(input)
        prob = num / np.sum(num, axis=1, keepdims=True)
        
        return prob


    def grads(self,X, pred, y, cache):
        n_X = pred.shape[0]
        y_yhat = pred
        y_yhat[range(n_X),y] -= 1
        dw2 = (cache.T).dot(y_yhat)
        db2 = np.sum(y_yhat, axis=0, keepdims=True)
        term_2 = y_yhat.dot(dw2.T) * (1 - np.power(cache,2))
        dw1 = np.dot(X.T, term_2)
        db1 = np.sum(term_2, axis=0)
        return (dw2,db2,dw1,db1)

    def train(self, train_set, va_set, lr=0.01, batch_size=100, n_epochs=10):
        train_data = train_set[0]
        train_labels = train_set[1]

        for epoch in range(n_epochs):
            # SGD
            for i in range(0, train_data.shape[0], batch_size):
                # Select batch.
                X = train_data[i:i + batch_size]
                y = train_labels[i:i + batch_size]
                n_X = X.shape[0]

                # Fwd prop.
                # We could have called forward() but let's save a function call !
                z1 = X.dot(self.w1) + self.b1
                a1 = self.activation(z1)
                z2 = a1.dot(self.w2) + self.b2
                pred = self.softmax(z2)

                # Back prop
                dw2,db2,dw1,db1 = self.grads(X, pred, y, a1)

                # Update params
                self.w1 -= lr * dw1
                self.b1 -= lr * db1
                self.w2 -= lr * dw2
                self.b2 -= lr * db2

    def test(self, test_set):
        return self.loss(test_set[0], test_set[1])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default=None, help='path to .npy data')

    args = parser.parse_args()
    train, valid, test = np.load(args.datafile)
    
    for hsize in range(80,160,5):
        print('# hidden :' , hsize)
        mlp = NN(hidden_dim= hsize, in_dim=784)
        mlp.initialize_weights(init_type = 'gauss')
        mlp.train(train, valid, 0.001, batch_size=10, n_epochs=10)
        print(mlp.test(test), mlp.num_params() / 1000000)

    #np.save('model.npy',(mlp.w1, mlp.b1, mlp.w2, mlp.b2))