#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:07:41 2017

@author: Vu Hoang Minh
"""

from __future__ import print_function

# import packages
from functools import partial
import os
import numpy as np

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K
from keras.utils import plot_model

# import load data
from data_handling_2d_patch import load_train_data, load_validatation_data
from train_main_2d_patch import get_unet_default, get_unet_reduced, get_unet_extended

# import configurations
import configs

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
image_type = configs.IMAGE_TYPE

# init configs
image_rows = configs.VOLUME_ROWS
image_cols = configs.VOLUME_COLS
image_depth = configs.VOLUME_DEPS
num_classes = configs.NUM_CLASSES

# patch extraction parameters
patch_size = configs.PATCH_SIZE
BASE = configs.BASE
smooth = configs.SMOOTH
nb_epochs  = configs.NUM_EPOCHS
batch_size  = configs.BATCH_SIZE
unet_model_type = configs.MODEL

# compute dsc
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# proposed loss function
def dice_coef_loss(y_true, y_pred):
    distance = 0
    for label_index in range(num_classes):
        dice_coef_class = dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])
        distance = 1 - dice_coef_class + distance
    return distance

# dsc per class
def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])

# get label dsc
def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def resume():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_gtruth_train = load_train_data()
    
    print('-'*30)
    print('Loading and preprocessing validation data...')
    print('-'*30)   
    imgs_val, imgs_gtruth_val  = load_validatation_data()
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    if unet_model_type == 'default':
        model = get_unet_default()
    elif unet_model_type == 'reduced':
        model = get_unet_reduced()
    elif unet_model_type == 'extended':
        model = get_unet_extended()       
        
    checkpoint_filepath_best = 'outputs/' + 'best_4classes_32_reduced_tuned_8915.h5'
    
    print(checkpoint_filepath_best)
    
    model.load_weights(checkpoint_filepath_best)        
    model.summary()
         
    print('-'*30)
    print('Fitting model...')
    print('-'*30)   
    #============================================================================
    print('training starting..')
    log_filename = 'outputs/' + image_type +'_model_train.csv' 
    #Callback that streams epoch results to a csv file.
    
    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
    
    #checkpoint_filepath = 'outputs/' + image_type +"_best_weight_model_{epoch:03d}_{val_loss:.4f}.hdf5"
    checkpoint_filepath = 'outputs/' + 'weights.h5'
    
    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    #callbacks_list = [csv_log, checkpoint]
    callbacks_list = [csv_log, early_stopping, checkpoint]

    #============================================================================
    hist = model.fit(imgs_train, imgs_gtruth_train, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, validation_data=(imgs_val,imgs_gtruth_val), shuffle=True, callbacks=callbacks_list) #              validation_split=0.2,
             
    model_name = 'outputs/' + image_type + '_model_last'
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'

# main	
if __name__ == '__main__':
    # folder to hold outputs
    if 'outputs' not in os.listdir(os.curdir):
        os.mkdir('outputs')
    resume()