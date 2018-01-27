#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:07:41 2017

@author: Vu Hoang Minh
"""

from __future__ import print_function

# import packages
from model import unet_model_3d
from keras.utils import plot_model
from keras import callbacks
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

# import load data
from data_handling import load_train_data, load_validatation_data

# import configurations
import configs

# init configs
patch_size = configs.PATCH_SIZE
batch_size = configs.BATCH_SIZE

config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape"] = (256, 128, 256)  # This determines what shape the images will be cropped/resampled to.
config["input_shape"] = (patch_size, patch_size, patch_size, 1)  # switch to None to train on the whole image (64, 64, 64) (64, 64, 64)
config["n_labels"] = 4
config["all_modalities"] = ['t1']#]["t1", "t1Gd", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["deconvolution"] = False  # if False, will use upsampling instead of deconvolution
config["batch_size"] = batch_size
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 31  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.0001
config["depth"] = configs.DEPTH
config["learning_rate_drop"] = 0.5

image_type = '3d_patches'

# 3D U-net depth=5
def generate_model(num_classes=4) :
    init_input = Input((1, 32, 32, 32))

    x = Conv3D(25, kernel_size=(3, 3, 3))(init_input)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)

    y = Conv3D(50, kernel_size=(3, 3, 3))(x)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)

    z = Conv3D(75, kernel_size=(3, 3, 3))(y)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)

    x_crop = Cropping3D(cropping=((6, 6), (6, 6), (6, 6)))(x)
    y_crop = Cropping3D(cropping=((3, 3), (3, 3), (3, 3)))(y)

    concat = concatenate([x_crop, y_crop, z], axis=1)

    fc = Conv3D(400, kernel_size=(1, 1, 1))(concat)
    fc = PReLU()(fc)
    fc = Conv3D(200, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)
    fc = Conv3D(150, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)

    pred = Conv3D(num_classes, kernel_size=(1, 1, 1))(fc)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, 9 * 9 * 9))(pred)
    pred = Permute((2, 1))(pred)
    pred = Activation('softmax')(pred)

    model = Model(inputs=init_input, outputs=pred)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])
    return model

# train
def train():
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

   # create a model
    model = unet_model_3d(input_shape=config["input_shape"],
                                depth=config["depth"],
                                pool_size=config["pool_size"],
                                n_labels=config["n_labels"],
                                initial_learning_rate=config["initial_learning_rate"],
                                deconvolution=config["deconvolution"])

    model.summary()
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    #============================================================================
    print('training starting..')
    log_filename = 'outputs/' + image_type +'_model_train.csv' 
    
    
    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
    
#    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
    
    #checkpoint_filepath = 'outputs/' + image_type +"_best_weight_model_{epoch:03d}_{val_loss:.4f}.hdf5"
    checkpoint_filepath = 'outputs/' + 'weights.h5'
    
    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    callbacks_list = [csv_log, checkpoint]
    callbacks_list.append(ReduceLROnPlateau(factor=config["learning_rate_drop"], patience=config["patience"],
                                           verbose=True))
    callbacks_list.append(EarlyStopping(verbose=True, patience=config["early_stop"]))

    #============================================================================
    hist = model.fit(imgs_train, imgs_gtruth_train, batch_size=config["batch_size"], nb_epoch=config["n_epochs"], verbose=1, validation_data=(imgs_val,imgs_gtruth_val), shuffle=True, callbacks=callbacks_list) #              validation_split=0.2,
        
     
    model_name = 'outputs/' + image_type + '_model_last'
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'

# main  
if __name__ == '__main__':
    train()