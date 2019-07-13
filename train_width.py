#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:57:18 2019

@author: soumak
"""

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#load data
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = ['0','1', '2', '3', '4', '5', '6', '7', '8','9']
print('Training image shape: ', train_images.shape)
print('Test image shape: ', test_images.shape)

#normalize the data
train_images = train_images / 255.0

test_images = test_images / 255.0

#reshape
train_images = train_images.reshape(-1, 28 * 28)
test_images = test_images.reshape(-1, 28 * 28)

#declare numpy array to hold 100 samples from each class, total 1000 records
train_sample = np.zeros((1000,train_images.shape[1]))
train_label_sample = np.zeros((1000,), dtype = np.uint8)

#select 100 samples from each class
# there are total 10 classes in mnist
j = 0
for class_num in class_names:
    count = 0
    for i in range(len(train_images)):
        if train_labels[i] == int(class_num) and count < 100:
            train_sample[j] =  train_images[i]
            train_label_sample[j] = train_labels[i]
            j += 1
            count += 1
            
#experiment with three layer network
def create_dense(widths):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Dense(units = widths, activation=tf.keras.activations.relu, input_shape=(784,), name='layer0'))
    model.add(keras.layers.Dense(units = widths, activation=tf.keras.activations.relu, name='layer1'))
    model.add(keras.layers.Dense(10, activation=tf.keras.activations.softmax, name='layer2'))
    return model

def evaluate(model, layers, batch_size=128, epochs=50):
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=False)
    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    training_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    model.save('model_depth'+str(layers)+'.h5')
    loss, accuracy  = model.evaluate(train_sample, train_label_sample, verbose=False)
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()
    print(f'Training loss: {training_loss[49]:.3}')
    print(f'Training accuracy: {acc[49]:.3}')
    print(f'Validation loss: {val_loss[49]:.3}')
    print(f'Validation accuracy: {val_acc[49]:.3}')

#This will 10 models with increasing depth each with 32 neurons
for layers in [1,2,4,8,16]:
    model = create_dense(32 * layers)
    evaluate(model, layers)


#model = create_model()
#model.summary()

#train the model
# include the epoch in the file name. (uses `str.format`)
#checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

#cp_callback = tf.keras.callbacks.ModelCheckpoint(
#    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
#    period=5)

#model = create_model()
#model.save_weights(checkpoint_path.format(epoch=0))
#model.fit(train_images, train_labels,
#          epochs = 50,
#          validation_data = (test_images,test_labels),
#          verbose=0)
#model.save('model.h5')