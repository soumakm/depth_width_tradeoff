#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:41:24 2019

@author: soumak
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

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
            
    
#calculate direction between first two points, x1-x2
t = np.arange(-3,10,0.5)
test_sample = np.zeros((len(t), train_sample.shape[1]))
for i in range(len(t)):
    x = train_sample[0] + t[i]*(train_sample[1] - train_sample[0])
    for j in range(len(x)):
        if x[i] < 0:
            x[i] = 0
        elif x[1] > 1:
            x[i] = 1
    test_sample[i] = x


#def create_model():
#  model = tf.keras.models.Sequential([
#    keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
#    keras.layers.Dropout(0.2),
#    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
#  ])
#  
#  model.compile(optimizer=tf.keras.optimizers.Adam(),
#                loss=tf.keras.losses.sparse_categorical_crossentropy,
#                metrics=['accuracy'])
#  
#  return model
#
#checkpoint_path = "training_2/"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#model = create_model()
#model.summary()
#latest = tf.train.latest_checkpoint(checkpoint_dir)
#print(latest)

#model = create_model()
prediction_list = []    
for layers in range(1, 10):
    model = keras.models.load_model('model'+str(layers)+'.h5')
    predictions = model.predict(test_sample)
    actual_prediction = np.argmax(predictions, axis=1)
    prediction_list.append(actual_prediction)
    #get_layer_before_softmax_output = K.function([model.layers[0].input], [model.layers[layers].output])
    #layer_output = get_layer_before_softmax_output([x])[0]
    #layer_name = 'layer'+str(layers - 1)
    #intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    #intermediate_output = intermediate_layer_model.predict(test_sample)
    output = model.layers[layers - 1].output
    output2 = Dense(10, name='newoutput')(output)
    model2 = Model(inputs = model.inputs, outputs=output2)

    wgts = model.layers[layers].get_weights()
    model2.get_layer('newoutput').set_weights(wgts)
    model3 = Model(inputs=model2.input, outputs=model2.get_layer("newoutput").output)
    intermediate_output = model3.predict(test_sample)
    for i in range(10):
        plt.plot(t, intermediate_output[:,i], label='class '+str(i))
        plt.legend(loc='best')
    plt.title('Generalization w.r.t Hidden Layer '+str(layers))
    plt.ylabel('Input to Softmax Layer')
    plt.xlabel('t')
    plt.show()

#model.load_weights(latest)
#loss, acc = model.evaluate(test_images, test_labels)
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))


#check the prediction

#print(np.argmax(predictions[0]))

