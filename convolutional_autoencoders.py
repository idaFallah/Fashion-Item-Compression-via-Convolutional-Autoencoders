
# importing libs

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
tf.__version__

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train.shape, y_train.shape

x_test.shape, y_test.shape

# visualizing images

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
width = 10
height = 10
fig, axes = plt.subplots(height, width, figsize= (15 , 15))
axes = axes.ravel()
for i in np.arange(0, width * height):
  index = np.random.randint(0, 60000)
  axes[i].imshow(x_train[index], cmap='gray')
  axes[i].set_title(classes[y_train[index]], fontsize = 8)
  axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)

# preprocessing the images

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

x_train.shape, x_test.shape

# building & training the Conv AE

autoencoder = Sequential()

# Encoder
autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), padding='same'))


autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2)))
autoencoder.add(Flatten())

autoencoder.summary()

# Decoder
autoencoder.add(Reshape((4, 4, 8)))

autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='valid'))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same'))

autoencoder.summary()

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(x_train, x_train, epochs=50)

# output = (input - filter + 1) / stride

# encoding & decoding the test images

autoencoder.summary()

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_14').output)
encoder.summary()

coded_test_images = encoder.predict(x_test)

coded_test_images.shape

decoded_test_images = autoencoder.predict(x_test)
decoded_test_images.shape

n_images = 10
test_images = np.random.randint(0, x_test.shape[0], size = n_images)
plt.figure(figsize=(18, 18))
for i, image_index in enumerate(test_images):
  # original images
  ax = plt.subplot(10, 10, i + 1)
  plt.imshow(x_test[image_index].reshape(28, 28), cmap='gray')
  plt.xticks(())
  plt.yticks(())

  # coded images
  ax = plt.subplot(10, 10, i + 1 + n_images)
  plt.imshow(coded_test_images[image_index].reshape(16, 8), cmap='gray')  # 16 * 8 = 128
  plt.xticks(())
  plt.yticks(())

  # decoded images
  ax = plt.subplot(10, 10, i + 1 + n_images*2)
  plt.imshow(decoded_test_images[image_index].reshape(28, 28), cmap='gray')
  plt.xticks(())
  plt.yticks(())











