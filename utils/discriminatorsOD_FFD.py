import keras

def buildDiscriminator1(columns):
    model = keras.models.Sequential()
    
    model.add(keras.layers.InputLayer(columns))
    
    model.add(keras.layers.Dense(512))
    
    model.add(keras.layers.Dense(256))

    model.add(keras.layers.Dense(1, activation="sigmoid"))

    return model