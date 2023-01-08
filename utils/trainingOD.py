import numpy as np

def trainOD(iterations, batch_size, generator, discriminator, network, data, columnCount):
    
    valid = np.ones((batch_size, 1, columnCount))
    generated = np.zeros((batch_size, 1, columnCount))
    
    for i in range(iterations):
        #train discriminator
        print(f"Iteration: {i}")

        #select audios from data
        indices = np.random.randint(0, len(data), batch_size)
        audios = data[indices]
        d_loss_real = discriminator.train_on_batch(audios, valid, return_dict=True)
        
        #get fake audio from generator
        noise = np.random.normal(0,0.05, (batch_size, 1, columnCount))  
        generated_audios = generator.predict(noise)
        d_loss_gen = discriminator.train_on_batch(generated_audios, generated, return_dict=True)
        
        print(f"d_loss_real: {d_loss_real}")
        print(f"d_loss_gen: {d_loss_gen}")
                            
        # train generator  
        noise = np.random.normal(0,0.05, (batch_size, 1, columnCount)) 
        g_loss = network.train_on_batch(noise, valid)

        print(f"g_loss: {g_loss}")                
                                  