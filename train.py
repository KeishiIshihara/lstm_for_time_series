#============================================================
# LSTM for time series dataset (household power consumption)
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
    parser.add_argument('-e','--epochs',default=10,type=int, help='Epoch')
    parser.add_argument('--test-size',default=0.1, help='Test size from 0 to 1. Default is 0.2')
    parser.add_argument('--not-record', '-nr', action='store_true', default=False)
    parser.add_argument('-t', '--time_steps',default=20, type=int, help='Time steps in each sequence data')
    args = parser.parse_args()

    #============================
    #         Configs
    #============================
    length_of_sequence = args.time_steps # time steps
    in_neurons = 7
    out_neurons = 1
    n_hidden = 100

    #============================
    #          Model
    #============================
    lstm = LSTM(n_hidden,
                activation='tanh',
                input_shape=(length_of_sequence, in_neurons),
                dropout=0.2,
                recurrent_dropout=0.2, 
                return_sequences=False
                )

    model = Sequential()
    model.add(lstm)
    model.add(Dense(units=out_neurons, activation='linear'))
    model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=0.001)
                #   metrics=['accuracy']
                  )
    model.summary()

    # Load data
    (X_train, X_test), (y_train, y_test) = load_time_series(debug=args.debug, 
                                                            test_size=args.test_size,
                                                            maxlen=length_of_sequence)

    #============================
    #         Callbacks
    #============================
    es_cb = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    callbacks = [es_cb]

    #============================
    #         Training
    #============================
    history = model.fit(X_train, y_train,
            batch_size=256,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1, # 2
            validation_split=0.2
            )

    #============================
    #         Prediction
    #============================
    X_test = X_test[500:700]
    y_test = y_test[500:700]
    predicted = model.predict(X_test)

    #=============================
    #     Draw training curves
    #=============================
    fig = plt.figure() # Initialize Figure
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(len(y_test)), y_test, color="blue", label="ground_truth")
    ax.plot(range(len(predicted)), predicted, color="red", label="predict_data")
    ax.legend(loc='best')
    ax.grid(which='both')
    ax.set_xlabel('Time')
    ax.set_ylabel('Global_active_power')
    fig.savefig('results/result.png')
    plt.close()
