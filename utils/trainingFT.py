import keras
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


def trainFT(iterations, batch_size, generator, discriminator, network, data, columnCount, discriminatorOptimizer,  Optimizer):
    
    
    
    valid = np.ones((batch_size, 1025, columnCount))
    generated = np.zeros((batch_size, 1025, columnCount), dtype=np.float32)
    noise2 = []
    d_loss_gen_accuracy = []
    d_loss_gen_loss = []
    d_loss_real_accuracy = []
    d_loss_real_loss = []
    g_loss2 = []
    g_loss_accuracy = []
    for i in range(iterations):

        print(f"Iteration: {i}")
        # select audios from data
        indices = np.random.randint(0, len(data), batch_size)
        audios = data[indices]
        now = datetime.datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H_%M")
        istftTestFile = "tm5g4d1gen-test-istft_"+date_time+".wav"
        glTestFile = "tm5g4d1gen-test-gl_"+date_time+".wav"
        
        
        # train discriminator with the "real" audio:
        print("---------------------TRAINIERE DISKRIMINATOR(REAL)---------------------")
        discriminator.trainable = True
        discriminator.compile(loss = "binary_crossentropy",
                      optimizer = discriminatorOptimizer,
                      metrics = [keras.metrics.Accuracy()])
        network.compile(loss="binary_crossentropy",
                       optimizer = Optimizer,
                       metrics = [keras.metrics.Accuracy()])
        
        # d_loss_real = discriminator.train_on_batch(audios, valid, return_dict=True)
        d_loss_real = discriminator.fit(audios, valid, batch_size, epochs = 2)
        print(f"g_loss: {d_loss_real.params}")
        print(d_loss_real.history['accuracy'])
        d_loss_real_accuracy.append(d_loss_real.history['accuracy'][len(d_loss_real.history['accuracy'])-1])
        d_loss_real_loss.append(d_loss_real.history['loss'][len(d_loss_real.history['loss'])-1])

        # get fake audio from generator
        print("---------------------TRAINIERE DISKRIMINATOR(GENERATED)---------------------")
        noise = np.random.normal(0,0.5,(batch_size, 1025, columnCount))
        noise2 = np.random.normal(0,0.5,(batch_size, 28, columnCount))
        generated_audios = generator.predict(noise)
        # d_loss_gen = discriminator.train_on_batch(generated_audios, generated, return_dict=True)
        print("Start Training...")
        d_loss_gen = discriminator.fit(generated_audios, generated, batch_size, epochs = 1)
        print(f"DISKRIMINATOR ACCURACY(GENERATED): {d_loss_gen.history['accuracy']}")
        d_loss_gen_accuracy.append(d_loss_gen.history['accuracy'][len(d_loss_gen.history['accuracy'])-1])
        d_loss_gen_loss.append(d_loss_gen.history['loss'][len(d_loss_gen.history['loss'])-1])

        #print(f"d_loss_real: {d_loss_real}")
        #print(f"d_loss_gen: {d_loss_gen}")
        
        discriminator.trainable = False
        print(discriminator.summary())
        discriminator.compile(loss = "binary_crossentropy",
                      optimizer = discriminatorOptimizer,
                      metrics = [keras.metrics.Accuracy()])
        network.compile(loss="binary_crossentropy",
                       optimizer = Optimizer,
                       metrics = [keras.metrics.Accuracy()])
        
        print(discriminator.trainable)
        print(discriminator.summary())

        # train generator
        print("---------------------TRAINIERE GAN---------------------")
        noise = np.random.normal(0,0.5,(batch_size, 1025, columnCount))
        noise2 = np.random.normal(0,0.5,(batch_size, 28, columnCount))
        
        print(f"NOISE: {noise2}")
        print(f"real Audio: {audios}")
        print(f"INDICES: {indices}")
        print(f"generated Audio:{generated_audios[0]}")
        
        glAudio = librosa.griffinlim(generated_audios[0])
        istftAudio = librosa.istft(generated_audios[0])
        ipd.Audio(data = glAudio, rate = 11025)
        ipd.Audio(data = istftAudio*100, rate = 11025)
        print(istftTestFile)
        sf.write(file = istftTestFile, data = istftAudio*100, samplerate = 11025)
        sf.write(file = glTestFile, data = glAudio, samplerate = 11025)
        
        # g_loss = network.train_on_batch(noise,valid, return_dict=True)
        g_loss = network.fit(noise,valid,batch_size, epochs = 5)
        g_loss2.append(g_loss.history['loss'][len(g_loss.history['loss'])-1])
        g_loss_accuracy.append(g_loss.history['accuracy'][len(g_loss.history['accuracy'])-1])

        print(f"GAN ACCURACY: {g_loss.history['accuracy']}")
    # history = network.fit(noise2, valid, epochs=iterations, batch_size=batch_size)
    
    print(f"d_loss_real Array: {d_loss_real_accuracy}")
    print(f"d_loss_gen Array: {d_loss_gen_accuracy}")

    noise2 = tf.convert_to_tensor(noise)
    print(f"Start plotting...")
    plt.plot(d_loss_real_accuracy, 'green')
    plt.plot(d_loss_gen_accuracy, 'red')
    plt.plot(g_loss_accuracy, 'blue')
    plt.title('GAN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['discriminator real', 'discriminator generated', 'gan'])
    plt.show()
    
    
    plt.plot(d_loss_real_loss, 'green')
    plt.plot(d_loss_gen_loss, 'red')
    plt.plot(g_loss2, 'blue')
    plt.title('GAN loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['discriminator real', 'discriminator generated', 'gan'])
    plt.show()

    #plt.plot([d['loss'] for d in g_loss2],'gan')
   # plt.title('GAN loss')
    #plt.ylabel('loss')
   # plt.xlabel('epoch')
   # plt.legend(['gan'])
   # plt.show()

def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


def trainMNIST(iterations, batch_size, discriminator, generator, gan, ):
    # example of loading the mnist dataset
    # load the images into memory
    (trainX, trainy), (testX, testy) = load_data()
    # summarize the shape of the dataset
    print('Train', trainX.shape, trainy.shape)
    print('Test', testX.shape, testy.shape)
    

              
              