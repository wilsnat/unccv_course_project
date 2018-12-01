import sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from easydict import EasyDict

import matplotlib.pyplot as plt
import pdb;
from data_loader import data_loader

dMode = "metrics_mode"

def main():
    data_out = import_and_prep_datasets(i_train_test_split = 0.9,  p_mode = dMode)
    print("train of shape: " + str(data_out.train.im.shape) + " test of shape: " + str(data_out.test.im.shape))
    data2_out = import_and_prep_datasets(i_train_test_split = 0.2, i_set = "set02", p_mode = dMode)

    all_data = combine_datasets(data_out,data2_out)
    model = build_model(all_data)
    model.summary()

    EPOCHS = 5000

    # Store training stats
    history = model.fit(all_data.train.full, all_data.train.y, epochs=EPOCHS,
                        verbose=0,
                        callbacks=[PrintDot()])
    plot_history(history)

    data_predictions = predict(model,data_out)
    print("")
    print("set01 pure data test mse: " + str(np.square(np.subtract(data_out.test.y,data_out.test.full[:,-3:])).mean()))
    print("set01 model test mse: " + str(np.square(np.subtract(data_out.test.y.flatten(), data_predictions)).mean()))
    plot(data_out.test.full[:,-3:],data_out.test.y).show()
    plot(data_out.test.y.flatten(), data_predictions).show()

    data2_predictions = predict(model,data2_out)
    print("set01 pure data test mse: " + str(np.square(np.subtract(data2_out.test.y,data2_out.test.full[:,-3:])).mean()))
    print("set01 model test mse: " + str(np.square(np.subtract(data2_out.test.y.flatten(), data2_predictions)).mean()))
    plot(data2_out.test.full[:,-3:],data2_out.test.y).show()
    plot(data2_out.test.y.flatten(), data2_predictions).show()

#intermediary function. you can just use data_loader() then data_prep() if you'd like
def import_and_prep_datasets(i_train_test_split = 0.7,
				i_input_image_size = (187, 250),
				i_data_path = 'dataset_full',
                i_set = "set01", p_mode = "center_mode",
                p_slice_mode_slice = 30):
    data = data_loader(train_test_split=i_train_test_split,
        input_image_size=i_input_image_size,
        data_path = i_data_path,
        set=i_set)
    return data_prep(data, mode=p_mode, slice_mode_slice=p_slice_mode_slice)

def combine_datasets(*datasets):
    if len(datasets) == 1:
        return datasets[0]
    else:
        all_data = EasyDict()
        all_data.train = EasyDict()
        all_data.test = EasyDict()
        all_data.train = datasets[0].train
        all_data.test = datasets[0].test
        for d in datasets[1:]:
            all_data.train.full = np.concatenate((all_data.train.full, d.train.full))
            all_data.train.y = np.concatenate((all_data.train.y, d.train.y))
            all_data.test.full = np.concatenate((all_data.test.full, d.test.full))
            all_data.test.y = np.concatenate((all_data.test.y, d.test.y))
        return all_data

def data_prep(data, mode = "center_mode", slice_mode_slice = 30):
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
    cH = round(h/2)
    cW = round(w/2)

    if mode == 'center_mode':
        data.train.immean = data.train.im.mean(axis=(1,2))
        data.test.immean = data.test.im.mean(axis=(1,2))
        data.train.im = data.train.im[:,cH-2:cH+3,cW-2:cW+3,:].mean(axis=(1,2))
        data.test.im = data.test.im[:,cH-2:cH+3,cW-2:cW+3,:].mean(axis=(1,2))

        data.train.full = np.concatenate((data.train.immean,data.train.im),axis=1)
        data.test.full = np.concatenate((data.test.immean,data.test.im), axis=1)
    elif mode == 'metrics_mode':
        data.train.immean = data.train.im.mean(axis=(1,2))
        data.test.immean = data.test.im.mean(axis=(1,2))
        data.train.immax = data.train.im.max(axis=(1,2))
        data.test.immax = data.test.im.max(axis=(1,2))
        data.train.immin = data.train.im.min(axis=(1,2))
        data.test.immin = data.test.im.min(axis=(1,2))
        data.train.im = data.train.im[:,cH-2:cH+3,cW-2:cW+3,:].mean(axis=(1,2))
        data.test.im = data.test.im[:,cH-2:cH+3,cW-2:cW+3,:].mean(axis=(1,2))

        data.train.full = np.concatenate((data.train.immean,data.train.immax,data.train.immin,data.train.im),axis=1)
        data.test.full = np.concatenate((data.test.immean,data.test.immax,data.test.immin,data.test.im), axis=1)
    elif mode == "slice_mode":
        sms = slice_mode_slice
        data.train.full = data.train.im[:,::sms,::sms,:]
        data.test.full = data.test.im[:,::sms,::sms,:].reshape(data.test.im.shape[0],-1)
        data.train.full = data.train.full.reshape(data.train.im.shape[0],-1)
        data.test.full = data.test.full.reshape(data.test.im.shape[0],-1)
        data.train.full = np.concatenate((data.train.full,data.train.im[:,cH,cW,:]),axis=1)
        data.test.full = np.concatenate((data.test.full,data.test.im[:,cH,cW,:]), axis=1)
    elif mode == "full_mode":
        data.train.full = data.train.im.reshape(data.train.im.shape[0],-1)
        data.test.full = data.test.im.reshape(data.test.im.shape[0],-1)
    data.train.full = np.concatenate((data.train.ex,data.train.full), axis=1)
    data.test.full = np.concatenate((data.test.ex,data.test.full), axis=1)
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
    if epoch % 300 == 0: print(str(epoch) + ": " + str(logs['loss']))
    print('.', end='')

#plot history of the training run (we could implement tensorboard later)
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['loss']),label = 'Val loss')
  plt.legend()
  plt.show()

#predict and show plot
def predict(model, data):
    test_predictions = model.predict(data.test.full).flatten()
    return test_predictions

#plot x,y. to show plot make sure to .show() the returned plot
#extra note: check to see if your model predictions are better than
# plot(data.test.im.center_pixel, data.test.y)
def plot(x,y):
    plt.title('data')
    plt.scatter(x, y)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100], [-100, 100])
    return plt

main()
