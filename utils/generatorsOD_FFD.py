import keras
from numpy import column_stack

#3x Dense
def buildGenerator1(columns):
    model = keras.models.Sequential()
    
    model.add(keras.layers.InputLayer(columns))
    
    model.add(keras.layers.Dense(256))

    model.add(keras.layers.Dense(512))

    model.add(keras.layers.Dense(1024))

    model.add(keras.layers.Dense(columns, activation="tanh"))
              
    return model