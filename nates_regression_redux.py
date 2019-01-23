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

#this sets the way the data is represented
dMode = "metrics_mode"
#this converts the data back to rgb before running the code if true
rgb_mode = True

#training variables
LEARNING_RATE = 0.0002
EPOCHS = 200
BATCH = 16
SHUFFLE = True

def main():
    """
    main control function
    """
    #import both data sets
    data_out = import_and_prep_datasets(i_train_test_split = 0.75,  p_mode = dMode, p_hue_augment=3)
    print("train of shape: " + str(data_out.train.im.shape) + " test of shape: " + str(data_out.test.im.shape))
    data2_out = import_and_prep_datasets(i_train_test_split = 0.65, i_set = "set02", p_mode = dMode, p_hue_augment = 7)

    all_data = combine_datasets(data_out,data2_out)

    #build model
    model = build_model(all_data,metrics_size=all_data.train.full.shape[1]-3)

    # Store training stats
    history = model.fit([all_data.train.full[:,:-3],all_data.train.full[:,-3:]], all_data.train.y, batch_size=BATCH,
                        epochs=EPOCHS,
                        verbose=0, shuffle=SHUFFLE,
                        callbacks=[PrintDot()])
    #graph error
    plot_history(history)

    #display colors and compare error of raw pixel value and ground truth/network output and ground truth
    data_predictions = model.predict([data_out.test.full[:,:-3],data_out.test.full[:,-3:]]).flatten()
    hail_mary = data_out.test.full[:,-3:]*(1-np.array([data_out.test.full[:,2]]).T+.5)
    print("")
    print("set01 center pixel test mse: " + str(np.power(np.subtract(data_out.test.y,data_out.test.full[:,-3:]),2).mean()))
    print("set01 model test mse: " + str(np.power(np.subtract(data_out.test.y.flatten(), data_predictions),2).mean()))
    print_colors(data_predictions.reshape(data_out.test.y.shape),data_out.test.full[:,-3:],data_out.test.y, "set01")

    data2_predictions = model.predict([data2_out.test.full[:,:-3],data2_out.test.full[:,-3:]]).flatten()
    hail_mary = data2_out.test.full[:,-3:]*((1-np.array([data2_out.test.full[:,2]]).T)*.8+.7)
    print("set02 center pixel test mse: " + str(np.power(np.subtract(data2_out.test.y,data2_out.test.full[:,-3:]),2).mean()))
    print("set02 model test mse: " + str(np.power(np.subtract(data2_out.test.y.flatten(), data2_predictions),2).mean()))
    print_colors(data2_predictions.reshape(data2_out.test.y.shape),data2_out.test.full[:,-3:],data2_out.test.y, "set02")

def import_and_prep_datasets(i_train_test_split = 0.7,
				i_input_image_size = (187, 250),
				i_data_path = 'dataset_full',
                i_set = "set01", p_mode = "center_mode",
                p_slice_mode_slice = 30, p_hue_augment=1):
    """
    Intermediary function. you can just use data_loader() then data_prep() if you'd like

    Parameters:
        i_ params: same params as inport
        p_ params: same params as data_prep

    Returns:
        data: same object out as data_prep
    """
    data = data_loader(train_test_split=i_train_test_split,
        input_image_size=i_input_image_size,
        data_path = i_data_path,
        set=i_set)
    return data_prep(data, mode=p_mode, slice_mode_slice=p_slice_mode_slice, hue_augment=p_hue_augment)

def combine_datasets(*datasets):
    """
    appends datasets

    Parameters:
        *datasets: any number of datasets to append

    Returns:
        data: appended dataset
    """
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
    """
    Takes image data and outputs metrics representing the image

    Number of different represntations are available

    Parameters:
        data: data downloaded with data_loader
        mode: type of data representation [center_mode,metrics_mode,slice_mode,full_mode]
        slice_mode_slice: size of the steps when sampling the image
        hue_argument: data augmentation based on hue shifts, values >1 shift the hue randomly hue_argument times

    Returns:
        data: same object as inputed but with the new subobjects train.full and test.full
            train.full: n representations of the training data in single numpy arrays
            test.full: n representations of the testing data in single numpy arrays
    """

    #iso scaling
    m=2400
    data.train.ex = np.sqrt((data.train.ex)/m)
    data.test.ex = np.sqrt((data.test.ex)/m)

    #hue augmentation
    if hue_augment>1:
        for hue_count in range(hue_augment-1):
            data.train.ex = np.concatenate((data.train.ex,data.train.ex),axis=0)
            holderim  = data.train.im
            holdery = data.train.y
            for i in range(data.train.im.shape[0]):
                offset = random.random()*360
                holderim[i,:,:,0] = (holderim[i,:,:,0]+offset)%360
                holdery[i,0] = (holdery[i,0]+offset)%360
            data.train.im = np.concatenate((data.train.im,holderim),axis=0)
            data.train.y = np.concatenate((data.train.y,holdery),axis=0)

    #after hue shift, allows for data to be represented in rgb mode
    if rgb_mode:
        for i in range(data.train.im.shape[0]):
            data.train.im[i] = (cv2.cvtColor(data.train.im[i], cv2.COLOR_HLS2RGB))
        data.train.y[:] =  (cv2.cvtColor(np.array([data.train.y[:]]), cv2.COLOR_HLS2RGB)).squeeze()
        for i in range(data.test.im.shape[0]):
            data.test.im[i] = (cv2.cvtColor(data.test.im[i], cv2.COLOR_HLS2RGB))
        data.test.y[:] =  (cv2.cvtColor(np.array([data.test.y[:]]), cv2.COLOR_HLS2RGB)).squeeze()
    else:
        #normalize hue value
        maxhue = 360
        data.train.im[:,:,:,0] = (data.train.im[:,:,:,0])/maxhue
        data.test.im[:,:,:,0] = (data.test.im[:,:,:,0])/maxhue
        data.train.y[:,0] = data.train.y[:,0]/maxhue
        data.test.y[:,0] = data.test.y[:,0]/maxhue
    h,w,d = data.train.im[0].shape
    cH = round(h/2)
    cW = round(w/2)

    #how the data is represented, based on global variable dMode
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

def build_model(data,metrics_size=1):
    """
    Keras sparse model

    Parameters:
        data: data.train data,
        metrics_size: length of the data input

    Returns:
        model: the full model
    """
    print(data.train.full.shape)
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
    #inputs
    im_metrics = keras.layers.Input([metrics_size])
    colors = keras.layers.Input([3])
    im_metrics_c = keras.layers.Dense(100, activation='relu')(im_metrics)
    im_metrics_coef = keras.layers.Dropout(.8)(im_metrics_c)
    #sparse layer
    H = keras.layers.Lambda(reshapeH_lam,[-1,1])([colors])
    L = keras.layers.Lambda(reshapeL_lam,[-1,1])([colors])
    S = keras.layers.Lambda(reshapeS_lam,[-1,1])([colors])
    im_metrics_H = keras.layers.Lambda(iso_lam,[-1,2])([im_metrics_coef,H])
    im_metrics_L = keras.layers.Lambda(iso_lam,[-1,2])([im_metrics_coef,L])
    im_metrics_S = keras.layers.Lambda(iso_lam,[-1,2])([im_metrics_coef,S])
    Hout =  keras.layers.Dense(1, activation=None)(im_metrics_H)
    Lout =  keras.layers.Dense(1, activation=None)(im_metrics_L)
    Sout =  keras.layers.Dense(1, activation=None)(im_metrics_S)
    #output
    yhat = keras.layers.Lambda(final_out,[-1,3])([Hout,Lout,Sout])
    out = keras.layers.Flatten()(yhat)
    model = keras.Model([im_metrics,colors],out)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])

    return model

###model lambda functions
def invmetrics_lam(iso):
    return 1-iso

def reshapeH_lam(input):
    i = 0
    colors = input[0]
    return keras.backend.reshape(keras.layers.Lambda(lambda x: x[:,i], output_shape=((1,)))(colors),[-1,1])

def reshapeL_lam(input):
    i = 1
    colors = input[0]
    return keras.backend.reshape(keras.layers.Lambda(lambda x: x[:,i], output_shape=((1,)))(colors),[-1,1])

def reshapeS_lam(input):
    i = 2
    colors = input[0]
    return keras.backend.reshape(keras.layers.Lambda(lambda x: x[:,i], output_shape=((1,)))(colors),[-1,1])

def iso_lam(input):
    isocoef = input[0]
    channel = input[1]
    return keras.layers.Concatenate(axis=1)([isocoef,channel])

def final_out(input):
    Hout = input[0]
    Lout = input[1]
    Sout = input[2]
    return keras.layers.Concatenate(axis=1)([Hout,Lout,Sout])
###

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    if epoch % 300 == 0: print(str(epoch) + "the epoch and the loss is " + str(logs['loss']))
    print('.', end='')

def plot_history(history):
    """
    Plot history of the training run

    Parameters:
        history: history from training
    """
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['loss']),label = 'Val loss')
    plt.legend()
    plt.show()

def plot(x,y):
    """
    Plot x,y

    Parameters:
        x, y: data
    """
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
    """
    print colors for color comparison

    Parameters:
        x, y, z: rgb or hsl colors
        title: graph title
    """
    plt.title(title)
    if rgb_mode:
        rgbx = (x*255).astype('int')
        rgby = (y*255).astype('int')
        rgbz = (z*255).astype('int')
    else:
        rgbx = x.copy()
        rgby = y.copy()
        rgbz = z.copy()
        rgbx[:,0] = rgbx[:,0]*360
        rgby[:,0] = rgby[:,0]*360
        rgbz[:,0] = rgbz[:,0]*360
        rgbx = (cv2.cvtColor(np.asarray([rgbx]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
        rgby = (cv2.cvtColor(np.asarray([rgby]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
        rgbz = (cv2.cvtColor(np.asarray([rgbz]), cv2.COLOR_HLS2RGB)*255).astype('int').squeeze()
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
