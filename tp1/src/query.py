#!/usr/bin/env python3

import numpy as np
from mlp import NN
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfile', type=str, default=None, help='path to .npy model')
    parser.add_argument('--datafile', type=str, default=None, help='path to .npy data')

    args = parser.parse_args()
    
    model = NN()
    model.w1, model.b1, model.w2, model.b2 , model.w3, model.b3 = np.load(args.modelfile)
    model.hidden_dims = (model.w2.shape[0], model.w2.shape[1])
    train, valid, test = np.load(args.datafile)

    index_elt = 1000
    p = 0
    g = model.check_grad( 
                            test[0][range(index_elt, index_elt + 1)], 
                            test[1][range(index_elt, index_elt+1)], 
                            (range(p, p + 1),range(p, p + 1)) , 100000)
 
    print(g)
    exit()
    grad_diff = np.zeros(10)
    stat_x = np.zeros(12)
    stat_y = np.zeros(12)
    l = 0

    for i in range(0,6):
        for k in [1,5]:
            epsilon = 1.0 / (k*np.power(10,i))

            for p in range(0,10):
                grad_diff[p] = model.check_grad( 
                            model.w3,
                            test[0][range(index_elt, index_elt + 1)], 
                            test[1][range(index_elt, index_elt+1)], 
                            (range(p, p + 1),range(p, p + 1)) , epsilon)
            print(i, k)
            stat_x[l] = epsilon
            stat_y[l] = grad_diff.max()
            l += 1

    print(stat_x, stat_y)
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(stat_x, stat_y,  label='max grad err')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.xlabel('epsilon', fontsize=10)
    plt.ylabel('error', fontsize=10)
    leg = ax.legend();
    fig.savefig('figure.png')   # save the figure to file
    plt.close(fig)    # close the figure
