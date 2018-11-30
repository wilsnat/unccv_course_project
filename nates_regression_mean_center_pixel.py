import sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

import matplotlib.pyplot as plt
import pdb;
from data_loader import data_loader

data = data_loader()

m = data.train.ex.max(axis=0)
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
pdb.set_trace()
data.train.im = data.train.im[:,round(h/2),round(w/2),:]
data.test.im = data.test.im[:,round(h/2),round(w/2),:]

data.train.full = np.concatenate((data.train.im,data.train.immean,data.train.ex), axis=1)
data.test.full = np.concatenate((data.test.im,data.test.immean,data.test.ex), axis=1)

print("train of shape: " + str(data.train.im.shape) + " test of shape: " + str(data.test.im.shape))

def build_model():
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

model = build_model()
model.summary()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1500

# Store training stats
history = model.fit(data.train.full, data.train.y, epochs=EPOCHS,
                    verbose=0,
                    callbacks=[PrintDot()])

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['loss']),label = 'Val loss')
  plt.legend()
  plt.show()

def predict(test_data = data.test.full):
    test_predictions = model.predict(test_data).flatten()

    plt.title('data')
    plt.scatter(data.test.y.flatten(), test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

plot_history(history)
predict()
