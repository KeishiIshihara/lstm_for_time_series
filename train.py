#============================================================
# LSTM for time series dataset (household power consumption)
#
# TODO: 
#  - picklize data
#  - training curve
# 
#    (c) Keishi Ishihara
#============================================================
from __future__ import print_function

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from load_csv_data import load_time_series

import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d','--debug',action='store_true',default=False, help='Execute with debug mode')
    parser.add_argument('--test-size',default=0.1, help='Test size from 0 to 1. Default is 0.2')
    parser.add_argument('--not-record', '-nr', action='store_true', default=False)
    parser.add_argument('--time_steps',default=20, help='Time steps in each sequence data')

    args = parser.parse_args()

    #============================
    #         Configs
    #============================
    logdir = './log' 

    length_of_sequence = args.time_steps # time steps
    in_neurons = 7
    out_neurons = 1
    n_hidden = 100

    #============================
    #          Model
    #============================
    lstm = LSTM(n_hidden,
                # batch_input_shape=(None, length_of_sequence, in_neurons),
                input_shape=(length_of_sequence, in_neurons),
                dropout=0.2,
                recurrent_dropout=0.2, 
                return_sequences=False)

    model = Sequential()
    model.add(lstm)
    model.add(Dense(units=out_neurons, activation='linear'))
    model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy'])
    model.summary()

    # Load data
    (X_train, X_test), (y_train, y_test) = load_time_series(debug=args.debug, 
                                                            test_size=args.test_size,
                                                            maxlen=args.time_steps)

    #============================
    #         Callbacks
    #============================
    if not args.not_record:
        # ms_cb = CILModelSave(args.output_dir_path, prefix) # for saving model
        # es_cb = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
        tb_cb = keras.callbacks.TensorBoard(log_dir=logdir)
        # callbacks = [ms_cb, tb_cb, es_cb]
        callbacks = [tb_cb]
    else:
        callbacks = None

    #============================
    #         Training
    #============================
    print(X_train.shape)
    for i in range(10):
        print('x_train {}, y_train {}'.format(X_train[i,:,0], y_train[i]))

    history = model.fit(X_train, y_train,
            batch_size=64,
            epochs=10,
            # callbacks=callbacks,
            validation_data=(X_test,y_test),
            verbose=1 # 2
            )

    #============================
    #         Evaluation
    #============================
    X_test = X_test[:20]
    y_test = y_test[:20]
    score_train = model.evaluate(x=X_train, y=y_train, verbose=1)
    score_test = model.evaluate(x=X_test, y=y_test, verbose=1)
    print('Train: loss={},accuracy={}'.format(score_train[0],score_train[1]))
    print('Test: loss={},accuracy={}'.format(score_test[0],score_test[1]))

    predicted = model.predict(X_test, batch_size=None, verbose=0, steps=None)
    print(X_test.shape)
    for i in range(len(predicted)):
        print('x_test {}, predict {}, y_test{}'.format(X_test[i,:,0], predicted[i], y_test[i]))

    #=============================
    #     Draw training curves
    #=============================
    fig = plt.figure() # Initialize Figure
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(len(predicted)), predicted, color="orange", label="predict_data")
    ax.plot(range(len(y_test)), y_test, color="blue", label="ground_truth")
    ax.plot(range(len(y_test)), X_test[:,:,0], color="green", label="previous_ts")
    ax.legend(loc='best')
    ax.set_xticks(range(0,len(X_test)-1))
    # ax.set_xticklabels()
    # ax.grid(which='both')
    ax.set_xlabel('time step')
    ax.set_ylabel('Global_active_power')
    fig.savefig('result.png')
    plt.close()
