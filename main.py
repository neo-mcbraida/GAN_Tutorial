from inspect import trace
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import PIL
from PIL import Image
import random

import matplotlib.pyplot as plt
import numpy as np
import os

# Set growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class GAN():
    def __init__(self):
        self.img_rows = 90#28
        self.img_cols = 90#28
        self.channels = 3#1
        self.batch_size = 32

        self.generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)

        self.folder_index = 0

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

        #self.path = "C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/images_small.npy"

        #self.path = "G:/AI DataSets/wikiart_dataset/wikiart/Art_Nouveau_Modern"
        #self.p2 = "G:/AI DataSets/wikiart_dataset/wikiart/Baroque"
        #self.path = "G:/AI DataSets/wikiart_dataset/wikiart/New_Realism"
        self.path = "G:/AI DataSets/faces/faces_6class/face_rococo"
        
        # self.path = "G:/AI DataSets/wikiart_dataset/wikiart/Post_Impressionism"
        #self.paths = [self.p1, self.p2, self.p3]

        
        self.data = []
       # self.load_data()

        
        self.noise_shape = (100)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = keras.optimizers.Adam(1.5e-3, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = layers.Input(shape=self.noise_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        #self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        #valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        #self.combined = keras.Model(z, valid)
        #self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        #self.load_dataset()

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    
    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def load_dataset(self):
        with open(self.paths, 'rb') as fp:
            self.data = np.load(fp)
        print(self.data[0].shape)
    
    @tf.function
    def train_step(self, images):
        seed = tf.random.normal([self.batch_size, 100])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(seed, training=True)
        
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def build_generator(self):

        model = keras.Sequential()
        
        model.add(layers.Dense(2*5*10,activation="relu",input_dim=self.noise_shape))

        model.add(layers.Dense(5*5*20,activation="relu"))
        #model.add(layers.BatchNormalization())
        #model.add(layers.LeakyReLU())

        model.add(layers.Dense(5*5*50,activation="relu"))
        #model.add(layers.BatchNormalization())
        #model.add(layers.LeakyReLU())

        model.add(layers.Dense(5*5*100,activation="relu"))
        #model.add(layers.Dense(5*5*100,activation="relu"))
        #model.add(layers.Dense(5*5*100,activation="relu"))
        #model.add(layers.BatchNormalization())
        #model.add(layers.LeakyReLU())

        model.add(layers.Dense(5*5*256,activation="relu"))
        #model.add(layers.BatchNormalization())
        #model.add(layers.LeakyReLU())

        
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((5, 5, 256)))# was (4, 4, 256)
        
        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(3, 3), padding='same', use_bias=False))#was (2, 2)
        assert model.output_shape == (None, 15, 15, 256) # was (8, 8)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(3, 3), padding='same', use_bias=False))
        assert model.output_shape == (None, 45, 45, 256) # was 30
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        
        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 90, 90, 256) # was (32, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 90, 90, 3)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 90, 90, 3)
        
        
        # model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        # assert model.output_shape == (None, 180, 180, 64) # was (96, 96)
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        # model.add(layers.BatchNormalization())
        assert model.output_shape == (None, 90, 90, 3)
        
        for layer in model.layers:
            print(layer.output_shape)

        model.summary()

        noise = layers.Input(shape=self.noise_shape)
        img = model(noise)

        return keras.Model(noise, img)

    def build_discriminator(self):
        model = keras.Sequential()

        model.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation="sigmoid"))

        # img_shape = (self.img_rows, self.img_cols, self.channels)

        # model = keras.Sequential()

        # model.add(layers.Flatten(input_shape=img_shape))
        # # model.add(layers.Rescaling(1./255))
        # model.add(layers.Dense(512))
        # model.add(layers.LeakyReLU(alpha=0.2))
        # model.add(layers.Dense(512))
        # model.add(layers.LeakyReLU(alpha=0.2))
        # model.add(layers.Dense(512))
        # model.add(layers.LeakyReLU(alpha=0.2))
        # model.add(layers.Dense(512))
        # model.add(layers.LeakyReLU(alpha=0.2))
        # model.add(layers.Dense(256))
        # model.add(layers.LeakyReLU(alpha=0.2))
        # model.add(layers.Dense(1, activation='sigmoid'))
        # model.summary()

        img = layers.Input(shape=self.img_shape)
        validity = model(img)

        return keras.Model(img, validity)

    def train(self, epochs, batch_size=64, save_interval=5):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)
        #dir_index = 0

        self.fixed_noise = np.random.normal(0, 1, (1, self.noise_shape))

        for epoch in range(epochs):
            
            # imgs = self.load_data(self.paths[dir_index], half_batch)
            # if imgs == False:
            #     dir_index += 1
            #     self.folder_index = 0
            # elif dir_index >= len(self.paths):
            #     print("trained successfully")
            #     break
            # else:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            ###############
            start_ind = 0
            end_ind = half_batch

            gen_loss = []
            disc_loss = []

            #random.shuffle(self.data)

            while end_ind < len(self.data):
                batch = self.data[start_ind:end_ind]
                batch = self.create_batch(batch)
                batch = batch/np.array(255)
                #noise = np.random.normal(0, 1, (half_batch, self.noise_shape))

                t = self.train_step(batch)
                gen_loss.append(t[0])
                disc_loss.append(t[1])

                start_ind += half_batch
                end_ind += half_batch

            gen_loss_av = sum(gen_loss) / len(gen_loss)
            disc_loss_av = sum(disc_loss) / len(disc_loss)


            print("Epoch: {}, gen loss={}, disc loss={}".format(epoch+1, gen_loss_av, disc_loss_av))
            ###############

            # # Select a random half batch of images
            # idx = np.random.randint(0, len(self.data), half_batch)
            # idx = idx.tolist()
            # mapping = map(self.data.__getitem__, idx)
            # imgs = list(mapping)#self.data[idx]# X_train[idx]
            # imgs = self.create_batch(imgs)
            # imgs = imgs/np.array(255)
            # noise = np.random.normal(0, 1, (half_batch, self.noise_shape))
            # # Generate a half batch of new images
            # gen_imgs = self.generator.predict(noise)

            # # Train the discriminator
            # d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # # ---------------------
            # #  Train Generator
            # # ---------------------

            # noise = np.random.normal(0, 1, (batch_size, self.noise_shape))

            # # The generator wants the discriminator to label the generated samples
            # # as valid (ones)
            # valid_y = np.array([1] * batch_size)

            # # Train the generator
            # g_loss = self.combined.train_on_batch(noise, valid_y)

            # # Plot the progress
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

            if epoch%1000 ==0:   
                self.discriminator.save('C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/discrimator.h5')
                self.generator.save('C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/generator.h5')

        self.discriminator.save('C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/discrimator.h5')
        self.generator.save('C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/generator.h5')

    def create_batch(self, images):
        batch_imgs = []
        for img in images:
            rgb_img = np.array(img)
            batch_imgs.append(rgb_img)
            #self.create_images([rgb_img/np.array(255)], "C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/art_images/mnist_test.png")#######
        return np.array(batch_imgs)

    def load_data(self):
        i = 0
        for filename in os.listdir(self.path):
            try:
                im_path = self.path + "/" + filename
                img = Image.open(im_path)
                img = img.resize((self.img_rows, self.img_cols), Image.ANTIALIAS)
                self.data.append(img)
                #img.close()
                i += 1
                print("image no." , i)
            except:
                print("not valid image")
        #self.data = np.array(self.data)

    def save_imgs(self, epoch):
        #r, c = 5, 5
        #noise = np.random.normal(0, 1, (r * c, 100))
        #noise = np.random.normal(0, 1, (1, 100))
        gen_imgs = self.generator(self.fixed_noise, training=False)

        path = "C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/art_images/mnist_%d.png" % epoch
        self.create_images(gen_imgs, path)

    def create_images(self, gen_images, path):
        #for image in gen_images:
        #    image *= 256
        img = gen_images[0]
        img = np.ascontiguousarray(img)
        img = (img * np.array(255)).astype(np.uint8) # needed as uint8..
        img = PIL.Image.fromarray(img, 'RGB')
        img.save(path)

    def load_model(self):
        self.generator = tf.keras.models.load_model('C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/generator.h5')

    def save_test_ims(self):
        for i in range(32):
            noise = np.random.normal(0, 1, (1, self.noise_shape))
            path = "C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/final_images/mnist_%d.png" % i
            img = self.generator(noise, training=False)
            self.create_images(img, path)

if __name__ == '__main__':
    gan = GAN()
    #gan.train(epochs=3000, batch_size=32, save_interval=2)
    gan.load_model()
    gan.save_test_ims()