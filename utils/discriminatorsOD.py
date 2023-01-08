import keras

#1x Dense + 1x LSTM
def buildDiscriminator1(columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(1, columns)))
    
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dense(256))

    model.add(keras.layers.Dense(columns, activation="sigmoid"))

    return model

#1x Dense + 1x LSTM
def buildDiscriminator2(columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(1, columns)))
    
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.LSTM(256, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="sigmoid"))

    return model

#1x Dense + 1x LSTM
def buildDiscriminator3(columns):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(1, columns)))

    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(keras.layers.LSTM(256, return_sequences=True))

    model.add(keras.layers.Dense(columns, activation="sigmoid"))

    return model