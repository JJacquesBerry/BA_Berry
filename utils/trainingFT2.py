import keras
import numpy as np
from keras import Model as model
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display as ipd
import librosa
import soundfile as sf
import datetime
import os


def trainFT2(iterations, batch_size, generator, discriminator, network, data, columnCount, discriminatorOptimizer,  Optimizer):
    
    
    
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
        
        # d_loss_real = discriminator.train_on_batch(audios, valid, return_dict=True)
        d_loss_real = discriminator.fit(audios, valid, batch_size, epochs = 2)
        print(f"g_loss: {d_loss_real.params}")
        print(d_loss_real.history['accuracy'])
        d_loss_real_accuracy.append(d_loss_real.history['accuracy'][len(d_loss_real.history['accuracy'])-1])
        d_loss_real_loss.append(d_loss_real.history['loss'][len(d_loss_real.history['loss'])-1])

        # get fake audio from generator
        print("---------------------TRAINIERE DISKRIMINATOR(GENERATED)---------------------")
        noise = np.random.normal(0,0.05,(batch_size, 1025, columnCount))
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
        
        print(discriminator.trainable)
        print(discriminator.summary())

        # train generator
        print("---------------------TRAINIERE GAN---------------------")
        noise = np.random.normal(0,0.05,(batch_size, 1025, columnCount))
        
        print(f"generated Audio:{generated_audios[0].shape}")
        
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

    noise2 = tf.convert_to_tensor(noise2)
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


def trainGAN(gan, generator, discriminator, Iterations, BatchSize, x_train):
    
    d_loss, d_accuracy = trainDis(generator = generator,
                         discriminator = discriminator,
                         XTrain = x_train,
                         Iterations = Iterations)
    
    print("---------------------TRAINIERE GAN---------------------")
    discriminator.trainable = False
    g_metrics = [] 
    g_loss = [] 
    g_accuracy = [] 
    YReal = np.ones((BatchSize, 1025, 108))
    for i in range(Iterations):
        noise = np.random.normal(0,0.05,(BatchSize, 1025, 108))

        #g_metrics = gan.fit(noise,YReal, BatchSize, epochs = 1)
        #g_loss.append(g_metrics.history['loss'][len(g_metrics.history['loss'])-1])
        #g_accuracy.append(g_metrics.history['accuracy'][len(g_metrics.history['accuracy'])-1])
        
        g_metrics = gan.train_on_batch(noise,YReal, return_dict=True)
        g_loss.append(g_metrics['loss'])
        g_accuracy.append(g_metrics['accuracy'])
        print(f"ITERATION {i}: {g_metrics}")

    
    print(f"Start plotting...")
    plt.plot(d_accuracy, 'red')
    plt.plot(g_accuracy, 'blue')
    plt.title('GAN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['discriminator', 'gan'])
    plt.show()
    
    plt.plot(d_loss, 'red')
    plt.plot(g_loss, 'blue')
    plt.title('GAN loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['discriminator', 'gan'])
    plt.show()

    return g_metrics
    
def trainDis(generator, discriminator, XTrain, Iterations):
    print("---------------------TRAINIERE DISKRIMINATOR---------------------")
    discriminator.trainable = True
    d_metrics = []
    d_loss = []
    d_accuracy = []
    BatchSize = 128
    
    XTest, YTest = getTest(generator, XTrain, BatchSize)
    
    for i in range(Iterations):
        #d_metrics = discriminator.fit(XTest, YTest, BatchSize, epochs = 1)
        #d_loss.append(d_metrics.history['loss'][len(d_metrics.history['loss'])-1])
        #d_accuracy.append(d_metrics.history['accuracy'][len(d_metrics.history['accuracy'])-1])
        
        d_metrics = discriminator.train_on_batch(XTest, YTest, return_dict=True)
        d_loss.append(d_metrics['loss'])
        d_accuracy.append(d_metrics['accuracy'])
        
        print(f"d_loss at Iteration {i}: {d_metrics}")
        #print(f"d_accuracy at Iteration {i}: {d_accuracy}")
        #print(f"d_loss at Iteration {i}: {d_loss}")
        
    return d_loss, d_accuracy
    
def getTest(generator, XTrain, BatchSize):
    XTest = []
    YTest = []
    HalfBatch = int(BatchSize/2)
    noise = np.random.normal(0,0.05,(HalfBatch, 1025, 108))
    
    YFake = np.zeros((HalfBatch, 1025,108), dtype=np.float32)
    YReal = np.ones((HalfBatch, 1025,108))
    XFake = generator.predict(noise)
    for i in range(HalfBatch):
        YTest.append(YFake[i])
        YTest.append(YReal[i])
        indices = np.random.randint(0, len(XTrain))
        
        
        XTest.append(XFake[i])
        XReal = XTrain[indices]
        XTest.append(XReal)
    
    return np.array(XTest), np.array(YTest)
    