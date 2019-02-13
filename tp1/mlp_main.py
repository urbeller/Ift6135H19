#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mlp import NN

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default=None, help='path to .npy data')

    args = parser.parse_args()
    train, valid, test = np.load(args.datafile)

    '''
    ## Initialization plot.

    bs = 20
    ep = 10
    lr = 0.01
    hd = (500,200)
    runs = 10

    itype = 'glorot'
    stat = np.zeros((1, ep))
    print(itype)
    for i in range(1,runs+1,1):
        print('\tRun=', i)
        mlp = NN(hidden_dims= hd, in_dim=784)
        mlp.initialize_weights(init_type = itype)
        gs = mlp.train(train, valid, lr, batch_size=bs, n_epochs=ep)
        stat += gs
    
    glorot_stat = stat / runs

    itype = 'gauss'
    stat = np.zeros((1, ep))
    print(itype)
    for i in range(1,runs+1,1):
        print('\tRun=', i)
        mlp = NN(hidden_dims= hd, in_dim=784)
        mlp.initialize_weights(init_type = itype)
        gs = mlp.train(train, valid, lr, batch_size=bs, n_epochs=ep)
        stat += gs
    
    gauss_stat = stat / runs

    itype = 'zeros'
    stat = np.zeros((1, ep))
    print(itype)
    for i in range(1,runs+1,1):
        print('\tRun=', i)
        mlp = NN(hidden_dims= hd, in_dim=784)
        mlp.initialize_weights(init_type = itype)
        gs = mlp.train(train, valid, lr, batch_size=bs, n_epochs=ep)
        stat += gs
    
    zeros_stat = stat / runs

    np.save('stats.npy',(glorot_stat, gauss_stat, zeros_stat))

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(range(1,ep+1), glorot_stat[0],  label='glorot')
    ax.plot(range(1,ep+1), gauss_stat[0],  label = 'gauss')
    ax.plot(range(1,ep+1), zeros_stat[0], label='zeros')
    leg = ax.legend();
    fig.savefig('init_comp.png')   # save the figure to file
    plt.close(fig)    # close the figure
    '''



    '''
    mlp = NN(hidden_dims= (500,200), in_dim=784)
    mlp.initialize_weights(init_type = 'glorot')
    print('#params: ', mlp.num_params() / 1000000.0)
    mlp.train(train, valid, 0.01, batch_size=10, n_epochs=10)
    print('Validation error: ', mlp.test(valid) )
    '''

    
    mlp = NN(hidden_dims= (600,100), in_dim=784)
    mlp.initialize_weights(init_type = 'glorot')
    print('#params: ', mlp.num_params() / 1000000.0)
    mlp.train(train, valid, 0.005, batch_size=10, n_epochs=10)
    print('Validation error: ', mlp.test(valid) )

    print('Test error: ', mlp.test(test), 'Accuracy : ', mlp.accuracy(test) * 100)
    #np.save('model.npy',(mlp.w1, mlp.b1, mlp.w2, mlp.b2, mlp.w3, mlp.b3))
    
