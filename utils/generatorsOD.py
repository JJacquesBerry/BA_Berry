import keras
from numpy import column_stack

#1x Dense + 1x LSTM
def buildGenerator1(columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(1, columns)))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.LSTM(512, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))
              
    return model

#2x LSTM
def buildGenerator2_1(columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(1, columns)))
    
    model.add(keras.layers.LSTM(256, return_sequences=True))
    model.add(keras.layers.LSTM(512, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))
              
    return model

#3x LSTM
def buildGenerator2_2(columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(1, columns)))
    
    model.add(keras.layers.LSTM(256, return_sequences=True))
    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(keras.layers.LSTM(1024, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))
              
    return model

#2x Dense + 1x LSTM
def buildGenerator3(columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(1, columns)))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.LSTM(1024, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))
              
    return model

#3x Dense + 1x LSTM
def buildGenerator4(columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(1, columns)))

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.LSTM(2048, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="tanh"))
              
    return model