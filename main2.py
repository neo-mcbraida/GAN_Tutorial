import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

from google.colab import drive, files

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

drive.mount("/content/gdrive")

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
        self.img_rows = 180#28
        self.img_cols = 180#28
        self.channels = 3#1
        self.batch_size = 32

        self.folder_index = 0

        #self.path = "C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/images_small.npy"

        self.p1 = "G:/AI DataSets/wikiart_dataset/wikiart/Art_Nouveau_Modern"
        self.p2 = "G:/AI DataSets/wikiart_dataset/wikiart/Baroque"
        self.p3 = "G:/AI DataSets/wikiart_dataset/wikiart/Realism"
        self.paths = [self.p1, self.p2, self.p3]
        
        self.noise_shape = (100)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = keras.optimizers.Adam(0.002, 0.5)

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
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        #self.load_dataset()
        self.data = []
        self.load_drive_data()


    def load_dataset(self):
        with open(self.paths, 'rb') as fp:
            self.data = np.load(fp)
        print(self.data[0].shape)

    def build_generator(self):

        model = keras.Sequential()
        
        model.add(layers.Dense(5*5*256,activation="relu",input_dim=self.noise_shape))
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
        
        
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 180, 180, 64) # was (96, 96)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 180, 180, 3)
        
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

    def train(self, epochs, batch_size=64, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)
        dir_index = 0

        for epoch in range(epochs):

            #imgs = self.load_data(self.paths[dir_index], half_batch)
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

            # Select a random half batch of images
            idx = np.random.randint(0, self.data.shape[0], half_batch)
            imgs = self.data[idx]# X_train[idx]
            imgs = imgs/np.array(255)
            noise = np.random.normal(0, 1, (half_batch, self.noise_shape))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.noise_shape))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def load_drive_data(self):
        set1 = []
        set2 = []
        with open("/content/gdrive/My Drive/train_set1.npy", 'rb') as fp:
            set1 = np.load(fp)
        with open("/content/gdrive/My Drive/train_set2.npy", 'rb') as fp:
            set2 = np.load(fp)
        self.data = np.concatenate((set1, set2), axis=None)
        print(self.data[0].shape)

    def get_data(self):
        try:
            path = direct + "/" + filename
            #rgb_im = imread(path)
            img = Image.open(path)
            #print(dim.shape)
            rgb_im = img.resize((width, height), Image.ANTIALIAS)
            rgb_im = np.array(rgb_im)
            #print(rgb_im.shape)
            images.append(rgb_im)
            img.close()
            self.data.append(images)
        except:
            print("not valid image")

    def load_data(self, path, half_batch):
        images = []
        batch_index = 0
        current_index = 0
        for filename in os.listdir(path):
            try:
                if current_index >= self.folder_index:
                    im_path = path + "/" + filename
                    img = Image.open(im_path)
                    rgb_im = img.resize((self.img_rows, self.img_cols), Image.ANTIALIAS)
                    rgb_im = np.array(rgb_im)
                    images.append(rgb_im)
                    img.close()
                    batch_index += 1
                    self.folder_index += 1
                    if batch_index > half_batch-1:
                        break
                current_index += 1
            except:
                print("not valid image")
        if batch_index < half_batch:
            return False
        else:
            return images

    def save_imgs(self, epoch):
        #r, c = 5, 5
        #noise = np.random.normal(0, 1, (r * c, 100))
        noise = np.random.normal(0, 1, (1, 100))
        gen_imgs = self.generator.predict(noise)

        path = "C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/art_images/mnist_%d.png" % epoch
        self.create_images(gen_imgs, path)

    def create_images(self, gen_images, path):
        for image in gen_images:
            image *= 256
        img = gen_images[0]
        img = np.ascontiguousarray(img)
        img = PIL.Image.fromarray(img, 'RGB')
        img.save(path)


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=25000, batch_size=32, save_interval=200)