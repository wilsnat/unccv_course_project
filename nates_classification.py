#I think there is a straight forward conversion from the color shifted by some percent of the iso.
#Maybe lower saturation some amount or shift it in one direction a bit
#I need to find a keras layer that connects a single value to all the other values

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
from classification_data_loader import data_loader

dMode = "metrics_mode"
#this converts the data back to rgb before running the code if true. I need to make an updated
# data_loader and model that doesn't ever play with hls.
hls_sucks = False

def main():
    data_out = import_and_prep_datasets(i_train_test_split = 0.7,  p_mode = dMode)
    print("train of shape: " + str(data_out.train.im.shape) + " test of shape: " + str(data_out.test.im.shape))
    #data2_out = import_and_prep_datasets(i_train_test_split = 0.5, i_set = "set02", p_mode = dMode, i_classificaiton_offset=32)

    all_data = data_out#combine_datasets(data_out,data2_out)
    model = build_model(all_data)

    EPOCHS = 100

    # Store training stats
    history = model.fit(all_data.train.full, all_data.train.ybin, batch_size=1,
                        epochs=EPOCHS,
                        verbose=0, shuffle=True,
                        callbacks=[PrintDot()])
    plot_history(history)

    test_loss, test_acc = model.evaluate(all_data.test.full, all_data.test.ybin,)
    #hail_mary = data_out.test.full[:,-3:]*(1-np.array([data_out.test.full]).T+.5)
    print("set01 loss: ", test_loss, " accuracy:", test_acc)
    #plot(data_out.test.full[:,-3:],data_out.test.y).show()
    #plot(data_out.test.y.flatten(), data_predictions).show()
    #print_colors(hail_mary,data_out.test.full[:,-3:],data_out.test.y, "hail mary")
    test_predictions = np.argmax(model.predict(all_data.test.full), axis=1)
    print_colors(data_out.test.y[test_predictions],data_out.test.full[:,-3:],data_out.test.y, "set01")

    #data2_predictions = predict(model,data2_out)
    #print("set02 hail_mary mae: " + str(np.abs(np.subtract(data2_out.test.y, hail_mary)).mean()))
    #plot(data2_out.test.full[:,-3:],data2_out.test.y).show()
    #plot(data2_out.test.y.flatten(), data2_predictions).show()
    #print_colors(hail_mary,data2_out.test.full[:,-3:],data2_out.test.y, "hail mary")
    #print_colors(data_out.test.y[data_predictions],data2_out.test.full[:,-3:],data2_out.test.y, "set02")
    #pdb.set_trace()

#intermediary function. you can just use data_loader() then data_prep() if you'd like
def import_and_prep_datasets(i_train_test_split = 0.7,
				i_input_image_size = (187, 250),
				i_data_path = 'dataset_full',
                i_set = "set01", p_mode = "center_mode",
                p_slice_mode_slice = 30,
                i_classificaiton_offset=0):
    data = data_loader(train_test_split=i_train_test_split,
        input_image_size=i_input_image_size,
        data_path = i_data_path,
        set=i_set,
        classification_offset = i_classificaiton_offset)
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
    #iso is exponential apparently
    data.train.ex = np.sqrt((data.train.ex)/m)
    data.test.ex = np.sqrt((data.test.ex)/m)
    if hls_sucks:
        for i in range(data.train.im.shape[0]):
            data.train.im[i] = (cv2.cvtColor(data.train.im[i], cv2.COLOR_HLS2RGB))
        data.train.y =  (cv2.cvtColor(np.array([data.train.y[1:]]), cv2.COLOR_HLS2RGB)).squeeze()
        for i in range(data.test.im.shape[0]):
            data.test.im[i] = (cv2.cvtColor(data.test.im[i], cv2.COLOR_HLS2RGB))
        data.test.y =  (cv2.cvtColor(np.array([data.test.y[1:]]), cv2.COLOR_HLS2RGB)).squeeze()
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
    data.train.ybin = np.zeros((data.train.yi.shape[0],32),dtype=int)
    for i,j in enumerate(data.train.yi):
        data.train.ybin[i,j] += 1
    print(data.train.ybin)
    data.test.ybin = np.zeros((data.test.yi.shape[0],32),dtype=int)
    for i,j in enumerate(data.test.yi):
        data.test.ybin[i,j] += 1
    print(data.test.ybin)
    data.train.full = np.concatenate((data.train.ex,data.train.full), axis=1)
    data.test.full = np.concatenate((data.test.ex,data.test.full), axis=1)
    return data

def build_model(data):
    print(data.train.full.shape)
        #relu dropout only for full/partial image?
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu,
                           input_shape=(data.train.full.shape[1],)),
       keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(32)
        ])
    optimizer = tf.train.AdamOptimizer()

    model.compile(loss=tf.losses.softmax_cross_entropy,
                optimizer=optimizer,
                metrics=['accuracy'])
    return model

def softmax_cross_entropy(y, pred): return tf.losses.softmax_cross_entropy(pred, y)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    if epoch % 300 == 0: print(str(epoch) + "th epoch and the loss is " + str(logs['loss']))
    print('.', end='')

#plot history of the training run (we could implement tensorboard later)
def plot_history(history):
  plt.figure()
  pdb.set_trace()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['acc']),label='Train accuracy')
  plt.plot(history.epoch, np.array(history.history['loss']),label = 'Train loss')
  plt.legend()
  plt.show()


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
