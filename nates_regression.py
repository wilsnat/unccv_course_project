import sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from easydict import EasyDict
import cv2
import random

import matplotlib.pyplot as plt
import pdb;
from data_loader import data_loader

dMode = "metrics_mode"
#this converts the data back to rgb before running the code if true. I need to make an updated
# data_loader and model that doesn't ever play with hls.
hls_sucks = False

def main():
    data_out = import_and_prep_datasets(i_train_test_split = 0.3,  p_mode = dMode)
    print("train of shape: " + str(data_out.train.im.shape) + " test of shape: " + str(data_out.test.im.shape))
    data2_out = import_and_prep_datasets(i_train_test_split = 0.5, i_set = "set02", p_mode = dMode)

    all_data = combine_datasets(data_out,data2_out)
    #iso is exponential apparently
    model = build_model(all_data)
    model.summary()

    EPOCHS = 500

    # Store training stats
    history = model.fit(all_data.train.full, all_data.train.y, batch_size=16,
                        epochs=EPOCHS,
                        verbose=0,
                        callbacks=[PrintDot()])
    plot_history(history)

    data_predictions = predict(model,data_out)

    print("")
    print("set01 center pixel test mae: " + str(np.abs(np.subtract(data_out.test.y,data_out.test.full[:,-3:])).mean()))
    print("set01 model test mae: " + str(np.abs(np.subtract(data_out.test.y.flatten(), data_predictions)).mean()))

    #plot(data_out.test.full[:,-3:],data_out.test.y).show()
    #plot(data_out.test.y.flatten(), data_predictions).show()

    print_colors(data_predictions.reshape(data_out.test.y.shape),data_out.test.full[:,-3:],data_out.test.y, "set01")

    data2_predictions = predict(model,data2_out)

    print("set02 center pixel test mae: " + str(np.abs(np.subtract(data2_out.test.y,data2_out.test.full[:,-3:])).mean()))
    print("set02 model test mae: " + str(np.abs(np.subtract(data2_out.test.y.flatten(), data2_predictions)).mean()))

    #plot(data2_out.test.full[:,-3:],data2_out.test.y).show()
    #plot(data2_out.test.y.flatten(), data2_predictions).show()

    print_colors(data2_predictions.reshape(data2_out.test.y.shape),data2_out.test.full[:,-3:],data2_out.test.y, "set02")

#intermediary function. you can just use data_loader() then data_prep() if you'd like
def import_and_prep_datasets(i_train_test_split = 0.7,
				i_input_image_size = (187, 250),
				i_data_path = 'dataset_full',
                i_set = "set01", p_mode = "center_mode",
                p_slice_mode_slice = 30, p_hue_augment=1):
    data = data_loader(train_test_split=i_train_test_split,
        input_image_size=i_input_image_size,
        data_path = i_data_path,
        set=i_set)
    return data_prep(data, mode=p_mode, slice_mode_slice=p_slice_mode_slice, hue_augment=p_hue_augment)

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

def data_prep(data, mode = "center_mode", slice_mode_slice = 20, hue_augment = 1):
    #m = np.concatenate((data.train.ex,  data.test.ex),axis=0).max(axis=0)

    m=2400
    #iso is exponential apparently
    data.train.ex = np.sqrt((data.train.ex)/m)
    data.test.ex = np.sqrt((data.test.ex)/m)
    if hue_augment>1:
        for hue_count in range(hue_augment-1):
            data.train.ex = np.concatenate((data.train.ex,data.train.ex),axis=0)
            holderim  = data.train.im
            holdery = data.train.y
            for i in range(data.train.im.shape[0]):
                offset = random.random()
                holderim[i,:,:,0] = (holderim[i,:,:,0]+offset)%1
                holdery[i,0] = (holdery[i,0]+offset)%1
            data.train.im = np.concatenate((data.train.im,holderim),axis=0)
            data.train.y = np.concatenate((data.train.y,holdery),axis=0)
    if hls_sucks:
        for i in range(data.train.im.shape[0]):
            data.train.im[i] = (cv2.cvtColor(data.train.im[i], cv2.COLOR_HLS2RGB))
        data.train.y[:] =  (cv2.cvtColor(np.array([data.train.y[:]]), cv2.COLOR_HLS2RGB)).squeeze()
        for i in range(data.test.im.shape[0]):
            data.test.im[i] = (cv2.cvtColor(data.test.im[i], cv2.COLOR_HLS2RGB))
        data.test.y[:] =  (cv2.cvtColor(np.array([data.test.y[:]]), cv2.COLOR_HLS2RGB)).squeeze()
    else:
        maxhue = 360
        data.train.im[:,:,:,0] = (data.train.im[:,:,:,0])/maxhue
        data.test.im[:,:,:,0] = (data.test.im[:,:,:,0])/maxhue
        data.train.y[:,0] = data.train.y[:,0]/maxhue
        data.test.y[:,0] = data.test.y[:,0]/maxhue
    h,w,d = data.train.im[0].shape
    cH = round(h/2)
    cW = round(w/2)

    #center pixel, image mean and exif
    if mode == 'center_mode':
        data.train.immean = data.train.im.mean(axis=(1,2))
        data.test.immean = data.test.im.mean(axis=(1,2))
        data.train.im = np.mean(data.train.im[:,cH-2:cH+3,cW-2:cW+3,:],axis=(1,2))
        data.test.im = np.mean(data.test.im[:,cH-2:cH+3,cW-2:cW+3,:],axis=(1,2))

        data.train.full = np.concatenate((data.train.immean,data.train.im),axis=1)
        data.test.full = np.concatenate((data.test.immean,data.test.im), axis=1)
    #center pixel, image mean, image max, image min and exif
    elif mode == 'metrics_mode':
        data.train.immean = np.mean(data.train.im,axis=(1,2))
        data.test.immean = np.mean(data.test.im,axis=(1,2))
        data.train.imstd = np.std(data.train.im,axis=(1,2))
        data.test.imstd = np.std(data.test.im,axis=(1,2))
        data.train.imvar = np.var(data.train.im,axis=(1,2))
        data.test.imvar = np.var(data.test.im,axis=(1,2))
        data.train.immax = data.train.im.max(axis=(1,2))
        data.test.immax = data.test.im.max(axis=(1,2))
        data.train.immin = data.train.im.min(axis=(1,2))
        data.test.immin = data.test.im.min(axis=(1,2))
        data.train.im = np.mean(data.train.im[:,cH-2:cH+3,cW-2:cW+3,:],axis=(1,2))
        data.test.im = np.mean(data.test.im[:,cH-2:cH+3,cW-2:cW+3,:],axis=(1,2))

        data.train.full = np.concatenate((data.train.immean,data.train.immax,data.train.immin,data.train.imstd,data.train.imvar,data.train.im),axis=1)
        data.test.full = np.concatenate((data.test.immean,data.test.immax,data.test.immin,data.test.imstd,data.test.imvar,data.test.im), axis=1)
    #center pixel, pixels every slice_mode_slice steps and exif
    elif mode == "slice_mode":
        sms = slice_mode_slice
        data.train.full = data.train.im[:,::sms,::sms,:].reshape(data.train.im.shape[0],-1)
        data.test.full = data.test.im[:,::sms,::sms,:].reshape(data.test.im.shape[0],-1)
        data.train.full = np.concatenate((data.train.full,data.train.im[:,cH,cW,:]),axis=1)
        data.test.full = np.concatenate((data.test.full,data.test.im[:,cH,cW,:]), axis=1)
    #cfull image and exif
    elif mode == "full_mode":
        data.train.full = data.train.im.reshape(data.train.im.shape[0],-1)
        data.test.full = data.test.im.reshape(data.test.im.shape[0],-1)
    data.train.full = np.concatenate((data.train.ex,data.train.full), axis=1)
    data.test.full = np.concatenate((data.test.ex,data.test.full), axis=1)
    return data

def build_model(data):
    print(data.train.full.shape)
        #relu dropout only for full/partial image?
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(data.train.full.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(3)
        ])
    optimizer = tf.train.RMSPropOptimizer(0.0002)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    if epoch % 300 == 0: print(str(epoch) + "th epoch and the loss is " + str(logs['loss']))
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

def print_colors(x,y,z,title="color comparison"):
    plt.title(title)
    if hls_sucks:
        rgbx = (x*255).astype('int')
        rgby = (y*255).astype('int')
        rgbz = (z*255).astype('int')
    else:
        rgbx = x.copy
        rgby = y.copy
        rgbz = z.copy
        rgbx[:,0] = rgbx[:,0]*360
        rgby[:,0] = rgby[:,0]*360
        rgbz[:,0] = rgbz[:,0]*360
        rgbx = (cv2.cvtColor(np.asarray([rgbx]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
        rgby = (cv2.cvtColor(np.asarray([rgby]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
        rgbz = (cv2.cvtColor(np.asarray([rgbz]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
    print(np.max(rgbx))
    palette = np.concatenate((rgbx,rgby,rgbz),axis=0)
    first_value = np.arange(0,rgbx.shape[0]-1)
    second_value = np.arange(rgbx.shape[0],2*rgbx.shape[0]-1)
    y_value = np.arange(2*rgbx.shape[0],3*rgbx.shape[0]-1)
    picture = np.concatenate(([y_value],[first_value],[second_value],[y_value]),axis=0)
    #this prints backwards on the graph, its actually y,yhat,pixel_data,y
    plt.ylabel('color([y],[pixel data],[yhat],[y])')
    plt.imshow(palette[picture])
    plt.show()


main()
