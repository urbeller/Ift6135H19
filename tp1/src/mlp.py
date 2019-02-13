#!/usr/bin/env python3

import numpy as np

class NN:
    def __init__(self,in_dim=784, hidden_dims=(100,100), out_dim=10):
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.input_dim = in_dim
        self.hidden_dims = hidden_dims
        self.output_dim = out_dim

    def num_params(self):
        n  = 1
        for a in (self.w1, self.w2, self.w3, self.b1, self.b2, self.b3):
            n += a.shape[0] * a.shape[1]
        return n

    def initialize_weights(self, init_type='zero'):
        self.b1 = np.zeros((1, self.hidden_dims[0]))
        self.b2 = np.zeros((1, self.hidden_dims[1]))
        self.b3 = np.zeros((1, self.output_dim))
    
        if init_type == 'gauss':
            self.w1 = np.random.randn(self.input_dim, self.hidden_dims[0])
            self.w2 = np.random.randn(self.hidden_dims[0], self.hidden_dims[1])
            self.w3 = np.random.randn(self.hidden_dims[1], self.output_dim)

        elif init_type == 'glorot':
            d = np.sqrt(6 / (self.input_dim + self.hidden_dims[0]))
            self.w1 = np.random.uniform(-d, d, (self.input_dim, self.hidden_dims[0]) )

            d = np.sqrt(6 / (self.hidden_dims[0] + self.hidden_dims[1]))
            self.w2 = np.random.uniform(-d, d, (self.hidden_dims[0], self.hidden_dims[1]) )

            d = np.sqrt(6 / (self.hidden_dims[1] + self.output_dim))
            self.w3 = np.random.uniform(-d, d, (self.hidden_dims[1], self.output_dim) )

        elif init_type == 'zeros':
            self.w1 = np.zeros((self.input_dim, self.hidden_dims[0]))
            self.w2 = np.zeros((self.hidden_dims[0], self.hidden_dims[1]))
            self.w3 = np.zeros((self.hidden_dims[1], self.output_dim))
        else :
            print('Unknown initialization type.')
            quit()
        
    def forward(self,input):
        z1 = input.dot(self.w1) + self.b1
        a1 = self.activation(z1)
        z2 = a1.dot(self.w2) + self.b2
        a2 = self.activation(z2)
        z3 = a2.dot(self.w3) + self.b3
        out_probs = self.softmax(z3)

        return (out_probs, a1, a2)

    def activation(self, x):
        return np.tanh(x)

    def loss(self,X, true_labels):
        data_size = X.shape[0]
        pred, a1, a2 = self.forward(X)
        logprobs = -np.log(pred[range(data_size),true_labels])

        return logprobs.mean()

    def softmax(self,input):
        num = np.exp(input - np.max(input))
        prob = num / np.sum(num, axis=1, keepdims=True)
        
        return prob


    def grads(self,X, pred, y, cache):
        a1, a2 = cache
        n_X = pred.shape[0]
        y_yhat = pred
        y_yhat[range(n_X),y] -= 1

        dw3 = (a2.T).dot(y_yhat)
        db3 = np.sum(y_yhat, axis=0, keepdims=True)

        term_2 = y_yhat.dot(self.w3.T) * (1 - np.power(a2,2))
        db2 = np.sum(term_2, axis=0)
        dw2 = np.dot(a1.T, term_2)

        term_1 =term_2.dot(self.w2.T) * (1 - np.power(a1,2))
        db1 = np.sum(term_1, axis=0)
        dw1 = np.dot(X.T, term_1)
 
        return (dw3, db3, dw2,db2,dw1,db1)

    def train(self, train_set, valid_set , lr=0.01, batch_size=100, n_epochs=10, log=False):
        stat = np.zeros((1, n_epochs))
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
                pred, a1, a2 = self.forward(X)

                # Back prop
                dw3,db3, dw2,db2,dw1,db1 = self.grads(X, pred, y, (a1,a2))

                # Update params
                self.w1 -= lr * dw1
                self.b1 -= lr * db1
                self.w2 -= lr * dw2
                self.b2 -= lr * db2
                self.w3 -= lr * dw3
                self.b3 -= lr * db3

                stat[0][epoch] = self.loss(X,y)

            if log:
                print('Epoch=', epoch + 1, self.loss(X,y), self.accuracy(valid_set))

        return stat
        
    def test(self, test_set):
        return self.loss(test_set[0], test_set[1])


    def check_grad(self, x, y, param_range, epsilon):
        print(self.w2[param_range])
        w2_bkp = np.copy(self.w2)
        w2p = np.copy(self.w2)
        w2n = np.copy(self.w2)

        w2p[param_range] += epsilon
        w2n[param_range] -= epsilon

        self.w2 = np.copy(w2p)
        loss_p = self.loss(x,y)

        self.w2 = np.copy(w2n)
        loss_n = self.loss(x,y)

        grad_n = ((loss_p - loss_n) / (2 * epsilon))

        self.w2 = np.copy(w2_bkp)

        pred, a1, a2 = self.forward(x)
        _, _, val, _, _, _ = self.grads(x, pred, y, (a1,a2))
        grad_a = val[param_range]

        return np.abs(grad_n - grad_a) 

    def accuracy(self, test_set):
        y_hat, a1, a2 = self.forward(test_set[0])
        pred_labels = np.argmax(y_hat, axis=1)
        correct_labels = (pred_labels==test_set[1]).sum()
        
        return correct_labels / test_set[0].shape[0]
