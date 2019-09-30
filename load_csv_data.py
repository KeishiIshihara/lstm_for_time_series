# coding: utf-8
#==========================
#  load data from csv
#
#   (c) Keishi Ishihara
#==========================
from __future__ import print_function

from sklearn.model_selection import train_test_split
import numpy as np
import csv
import sys

def load_time_series(filename='household_power_consumption.txt', test_size=0.2, maxlen=10, debug=False):
    '''
    ## Input: csv filename, test size
    ## Output: data and its label for training and testing
    ### Dataset specification:
     - data: 7 columns,  all attributes(t-1)
     - label: 1 column, Global_active_power(t) (GAP)

     * Note:This csv includes some '?' and ''(blank)
    '''

    print('Now loading csv file..')
    data = []
    with open(filename) as f:
        line = csv.reader(f,delimiter=';')
        next(line)
        if not debug:
            for row in line:
                data.append(row)
        else:
            i = 0
            for row in line:
                if i > 100000: break
                data.append(row)
                i += 1

    data = np.array(data)
    data = data[:, 2:] # delete non metric columns
    data[data == '?'] = 0 # replace non-metric values
    data[data == ''] = 0 # replace non-metric values
    data = data.astype(np.float16) # cast

    # Normalize for each column at here if necessary

    new_data, new_label = [], []
    for i in range(len(data) - maxlen):
        new_data.append(data[i:i + maxlen,:])
        new_label.append(data[i+maxlen,0])

    new_data = np.array(new_data).reshape(len(new_data), maxlen, 7)
    new_label = np.array(new_label).reshape(len(new_label), 1)

    # Split those data to train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_label, test_size=test_size, shuffle=False)

    print('Data specification:')
    print('  - (x_train, y_train) = ({}, {})'.format(len(X_train), len(y_train)))
    print('  - (x_test, y_test) = ({}, {})'.format(len(X_test), len(y_test)))
    print(X_train.shape)

    return (X_train, X_test), (y_train, y_test)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data loading program')
    parser.add_argument('-d','--debug',action='store_true',default=False, help='Execute with debug mode')
    parser.add_argument('--test-size',default=0.2, help='Test size from 0 to 1. Default is 0.2')
    parser.add_argument('--time_steps',default=20, help='Time steps in each sequence data')
    args = parser.parse_args() 

    print('Debug mode=',args.debug)
    print('Train:Test={}:{}'.format(1-args.test_size, args.test_size))
    (x_train, y_train), (x_test, y_test) = load_time_series(debug=args.debug, 
                                                            test_size=args.test_size,
                                                            maxlen=args.time_steps)

