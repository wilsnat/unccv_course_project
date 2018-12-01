import sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

import matplotlib.pyplot as plt
import pdb;
from data_loader import data_loader

def main():
    data = data_loader()
    data = data_prep(data)
    print("train of shape: " + str(data.train.im.shape) + " test of shape: " + str(data.test.im.shape))
    model = build_model(data)
    model.summary()

    EPOCHS = 1500

    # Store training stats
    history = model.fit(data.train.full, data.train.y, epochs=EPOCHS,
                        verbose=0,
                        callbacks=[PrintDot()])
    plot_history(history)

    data2 = data_loader(train_test_split = 0.04, set = "set02")
    data2 = data_prep(data2)

    #predict(model,data)
    predict(model,data2)

def data_prep(data):
    #m = np.concatenate((data.train.ex,  data.test.ex),axis=0).max(axis=0)
    m=2400
    data.train.ex = (data.train.ex)/m
    data.test.ex = (data.test.ex)/m
    maxhue = 360
    data.train.im[:,:,:,0] = (data.train.im[:,:,:,0])/maxhue
    data.test.im[:,:,:,0] = (data.test.im[:,:,:,0])/maxhue
    data.train.y[:,0] = data.train.y[:,0]/maxhue
    data.test.y[:,0] = data.test.y[:,0]/maxhue

    h,w,d = data.train.im[0].shape
    data.train.immean = data.train.im.mean(axis=(1,2))
    data.test.immean = data.test.im.mean(axis=(1,2))
    cH = round(h/2)
    cW = round(w/2)
    data.train.im = data.train.im[:,cH-3:cH+3,cW-3:cW+3,:].mean(axis=(1,2))
    data.test.im = data.test.im[:,cH-3:cH+3,cW-3:cW+3,:].mean(axis=(1,2))

    data.train.full = np.concatenate((data.train.im,data.train.immean,data.train.ex), axis=1)
    data.test.full = np.concatenate((data.test.im,data.test.immean,data.test.ex), axis=1)
    return data

def build_model(data):
    print(data.train.full.shape[1])
    model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(data.train.full.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(3)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['loss']),label = 'Val loss')
  plt.legend()
  plt.show()

def predict(model, data):
    pdb.set_trace()
    test_predictions = model.predict(data.test.full).flatten()

    plt.title('data')
    plt.scatter(data.test.y.flatten(), test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

main()
