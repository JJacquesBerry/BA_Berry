import keras

#
def buildGAN1(discriminator, generator):
    model = keras.models.Sequential([discriminator, generator])
    
    return model