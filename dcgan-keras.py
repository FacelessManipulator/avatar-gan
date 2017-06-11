from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import argparse
import math
from matplotlib import pyplot as plt

BATCH_SIZE = 32
EPOCH = 5

imageDataGenerator = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
#     shear_range=0.2,
#     zoom_range=0.2,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1./255,
    preprocessing_function=None,)
train_datagen = imageDataGenerator.flow_from_directory(
        'data',
        target_size=(75, 75),
        batch_size=BATCH_SIZE,
#         color_mode='rgb',
        class_mode='binary')

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024, activation='elu'))
    model.add(Dense(128*16*16, activation='elu'))
    model.add(Reshape((16, 16, 128)))
    model.add(BatchNormalization(axis=-1))
    model.add(UpSampling2D(size=(2, 2)))#增采样扩充，假装反池化
    model.add(Conv2DTranspose(64, 5, 5, activation='elu',kernel_initializer='truncated_normal'))
    model.add(BatchNormalization(axis=-1))#for channel_last
    model.add(UpSampling2D(size=(2, 2)))#增采样扩充，假装反池化
    model.add(Conv2DTranspose(3, 4, 4, activation='elu',kernel_initializer='truncated_normal'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, 4, 4,
                     activation='elu',
                     kernel_initializer='truncated_normal',
                    input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, 5, 5, 
                     activation='elu',
                     kernel_initializer='truncated_normal',))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024,activation='elu'))
    model.add(Dense(64,activation='elu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def train():
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    d_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    g_optim = Adam(lr=0.002, beta_1=0.5, beta_2=0.999)
    generator.compile(loss='binary_crossentropy', optimizer="Adam")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(EPOCH):
        for index,data in enumerate(train_datagen,0):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)#uniform noise for each instance.
            image_batch = data[0]*2-1
            generated_images = generator.predict(noise, verbose=0)
            X = np.concatenate((image_batch, generated_images))
#             print(image_batch[0], generated_images[0])
#             return 0
            y = np.concatenate([np.ones(image_batch.shape[0]),np.zeros(BATCH_SIZE)])
            d_loss = discriminator.train_on_batch(X, y)
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            if index % 20 == 0:
                print("[%d/%d] [%d/%d]"%(epoch,EPOCH, index, BATCH_SIZE))
                print("batch %d d_loss : %f" % (index, d_loss))
                print("batch %d g_loss : %f" % (index, g_loss))
            if index % 300 == 0:
                imgs_grid = np.concatenate([generated_images[i] for i in range(BATCH_SIZE)], axis=1)
                plt.imsave('gen/'+str(epoch)+"_"+str(index)+".png",(imgs_grid+1)/2)
            if index > 1700//BATCH_SIZE:
                break#break the loop manually
#             if index % 10 == 9:
#                 generator.save_weights('generator', True)
#                 discriminator.save_weights('discriminator', True)

if __name__ == '__main__':
    train()