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
config["early_stop"] = 30  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00005
config["depth"] = configs.DEPTH
config["learning_rate_drop"] = 0.5

image_type = '3d_patches'

# resume training
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

   # create a model
    model = unet_model_3d(input_shape=config["input_shape"],
                                depth=config["depth"],
                                pool_size=config["pool_size"],
                                n_labels=config["n_labels"],
                                initial_learning_rate=config["initial_learning_rate"],
                                deconvolution=config["deconvolution"])

    model.summary()
    
    checkpoint_filepath_best = 'outputs/' + 'best_weights_125extract_depth5_patch32_88_943_935.h5'
    checkpoint_filepath_best = 'outputs/' + 'best_weights_10extract_depth5_patch32_855_945_935.h5'
    checkpoint_filepath_best = 'outputs/' + 'best_weights_12extract_depth5_patch32_85_946_931_norm_tuned10.h5'
    checkpoint_filepath_best = 'outputs/' + 'best_weights.h5'
    #checkpoint_filepath_best = 'outputs/' + 'best_weights_11extract_depth5_patch32_855_945_935_tunedfrom10extract.h5'
    
    #checkpoint_filepath_best = 'outputs/' + 'best_weights_125extract_depth4_patch32_864_941_932.h5'
    model.load_weights(checkpoint_filepath_best)
    
    print('*'*50)
    print('Load model: ', checkpoint_filepath_best)
    print('*'*50)
    
    #summarize layers
    #print(model.summary())
    # plot graph
    #plot_model(model, to_file='3d_unet.png')
    
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

    
if __name__ == '__main__':
    resume()