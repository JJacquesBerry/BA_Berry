import keras.layers as layers
from keras.models import Sequential
import keras
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU

#1x Dense + 1x LSTM
def buildDiscriminator1(rows, columns):
    model = keras.models.Sequential()
    
    model.add(keras.layers.InputLayer(input_shape=(rows, columns)))

    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dense(256))

    model.add(keras.layers.Dense(columns, activation="sigmoid"))

    return model

#1x Dense + 1x LSTM
def buildDiscriminator2(rows, columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(rows, columns)))
    
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.LSTM(256, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="sigmoid"))

    return model

#1x Dense + 1x LSTM
def buildDiscriminator3(rows, columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(rows, columns)))
    
    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(keras.layers.LSTM(256, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="sigmoid"))

    return model

def buildDiscriminator4(rows, columns):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(rows, columns)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    #model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    return model