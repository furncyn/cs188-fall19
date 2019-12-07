# -*- coding: utf-8 -*-
"""cs188-hw3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T6yzH96n7PM9uyMEjPPZ_WBzPch9fwJh

# Import the Required Dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
# %matplotlib inline
import tensorflow as tf
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
import keras
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.regularizers import l1
import os
import time
from time import time

def plot_learningCurve(history,num_epoch):
  # Plot training & validation accuracy values
  epoch_range = np.arange(1,num_epoch+1)
  plt.plot(epoch_range, history.history['acc'])
  plt.plot(epoch_range, history.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

save_dir='/model/resnet'

def identity_block(X, f, filters):
    """
    Implementation of the identity block as defined in Figure 4 of homework

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path (e.g. [2,4,16])
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding="same", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X

"""##Testing the Identity Block

Simply run the code in the following block and report the result that is generated in your homework report.
"""

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [4, 4, 4, 8])
    X = np.random.randn(4,4,4,8)
    A = identity_block(A_prev, f=2, filters=[2, 4, 8])
    test.run(tf.global_variables_initializer())
    res = test.run([A], feed_dict={A_prev: X})
    print('Result = {}'.format(res[0][1][1][0]))


def convolutional_block(X, f, filters,stride=2):
    """
    Implementation of the convolutional block as defined in Figure 5 of homework

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path (e.g. [2,4,16])
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=f1, kernel_size=(1,1), strides=(stride,stride), padding="valid", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=f2, kernel_size=(f,f), strides=(1,1), padding="same", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters=f3, kernel_size=(1,1), strides=(1,1), padding="valid", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=f3, kernel_size=(1,1), strides=(stride,stride), padding="valid", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [4, 4, 4, 8])
    X = np.random.randn(4,4,4,8)
    A = convolutional_block(A_prev, f=2, filters=[2, 4, 8])
    test.run(tf.global_variables_initializer())
    res = test.run([A], feed_dict={A_prev: X})
    print('Result = {}'.format(res[0][1][1][0]))


def ResNet(input_shape=(32, 32, 3), classes=10):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D(padding=(3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="valid", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, filters=[64,64,256], f=3, stride=1)
    X = identity_block(X, filters=[64,64,256], f=3)
    X = identity_block(X, filters=[64,64,256], f=3)

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, filters=[128,128,512], f=3, stride=2)
    X = identity_block(X, filters=[128,128,512], f=3)
    X = identity_block(X, filters=[128,128,512], f=3)
    X = identity_block(X, filters=[128,128,512], f=3)

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, filters=[256,256,1024], f=3, stride=2)
    X = identity_block(X, filters=[256,256,1024], f=3)
    X = identity_block(X, filters=[256,256,1024], f=3)
    X = identity_block(X, filters=[256,256,1024], f=3)
    X = identity_block(X, filters=[256,256,1024], f=3)
    X = identity_block(X, filters=[256,256,1024], f=3)

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, filters=[512,512,2048], f=3, stride=2)
    X = identity_block(X, filters=[256,256,2048], f=3)
    X = identity_block(X, filters=[256,256,2048], f=3)

    # AVGPOOL (≈1 line)
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation="softmax")(X)


    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNe')
    return model

model = ResNet(input_shape=(32, 32, 3), classes=10)

"""##CIFAR10 Dataset

Simply run the following code block to download and preprocess the CIFAR10 dataset. We also use online data-augmentation to improve the results.
"""

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
num_classes = 10
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

datagen.fit(x_train)

"""##Compile the Model

The following block sets the required hyper-parameters, complies the model starts the process of training and at the end saves the trained model.

You can use your own hyper-parameters, but these have been tested to work properly.

Note that we require you to report the accuracy for models that have been trained for 50 epochs.
"""

batch_size = 2048
  epochs = 50
  data_augmentation = True
  learning_rate=0.001

  opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
  model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
  t1=time()
  history =model.fit_generator(datagen.flow(x_train, y_train,
                      batch_size=batch_size),
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      workers=4)
  print('Training time is {} s'.format(time()-t1))
  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)

  model_name='resnet'
  model_path = os.path.join(save_dir, model_name)
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)

"""##Evaluate the model

Simply run the following block of code to plot accuracies on training and validation set during different training epochs and eventually get the **accuracy** of the trained model on the **testing set** of CIFAR10 dataset
"""

plot_learningCurve(history,epochs)

# Evaluate trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
