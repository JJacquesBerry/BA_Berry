import keras
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.models import Sequential
import numpy as np
from keras import Model as model
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display as ipd
import librosa
import soundfile as sf
import datetime
import os

#1x Dense + 1x LSTM
def buildGenerator1(rows, columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(rows, columns)))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.LSTM(512, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))
    model.add(keras.layers.Reshape((rows, columns)))

    return model

#2x LSTM
def buildGenerator2_1(rows, columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(rows, columns)))

    model.add(keras.layers.LSTM(256, return_sequences=True))
    model.add(keras.layers.LSTM(512, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))
    model.add(keras.layers.Reshape((rows, columns)))

    return model   

#3x LSTM
def buildGenerator2_2(rows, columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(rows, columns)))

    model.add(keras.layers.LSTM(256, return_sequences=True))
    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(keras.layers.LSTM(1024, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))
    model.add(keras.layers.Reshape((rows, columns)))

    return model 

#2x Dense + 1x LSTM
def buildGenerator3(rows, columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(rows, columns)))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.LSTM(1024, return_sequences=True))
    

    model.add(keras.layers.Dense(columns, activation="tanh"))
    model.add(keras.layers.Reshape((rows, columns)))

    return model   


#3x Dense + 1x LSTM
def buildGenerator4(rows, columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(rows, columns)))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.LSTM(2048, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))

    model.add(keras.layers.Reshape((rows, columns)))

    return model

def buildGenerator5(rows, columns, latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # upsample to 14x14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(keras.layers.Dense(columns, activation="tanh"))
    model.add(keras.layers.Reshape((rows, columns)))
    return model