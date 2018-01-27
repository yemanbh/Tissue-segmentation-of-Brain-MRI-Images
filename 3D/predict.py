#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:07:41 2017

@author: Vu Hoang Minh
"""

from __future__ import print_function

# import packages
from functools import partial
import os, time
import numpy as np
from model import unet_model_3d
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K
from keras.utils import plot_model
import nibabel as nib
from PIL import Image
from sklearn.feature_extraction.image import extract_patches

# import load data
from data_handling import load_train_data, load_validatation_data

# import configurations
import configs

patch_size = configs.PATCH_SIZE
config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape"] = (256, 128, 256)  # This determines what shape the images will be cropped/resampled to.
config["input_shape"] = (patch_size, patch_size, patch_size, 1)  # switch to None to train on the whole image (64, 64, 64) (64, 64, 64)
config["n_labels"] = 4
config["all_modalities"] = ['t1']#]["t1", "t1Gd", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["deconvolution"] = False  # if False, will use upsampling instead of deconvolution
config["batch_size"] = 8
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 20  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.0001
config["depth"] = configs.DEPTH
config["learning_rate_drop"] = 0.5

extraction_reconstruct_step = configs.EXTRACTION_RECONSTRUCT_STEP
image_type = '3d_patches'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

image_type = configs.IMAGE_TYPE

# init configs
image_rows = configs.VOLUME_ROWS
image_cols = configs.VOLUME_COLS
image_depth = configs.VOLUME_DEPS
num_classes = configs.NUM_CLASSES

# patch extraction parameters

BASE = configs.BASE
smooth = configs.SMOOTH
nb_epochs  = configs.NUM_EPOCHS
batch_size  = configs.BATCH_SIZE
unet_model_type = configs.MODEL
extraction_step = 1

train_imgs_path = '../data_new/Validation_Set'
checkpoint_filename = 'best_3classes_32_85_93_92_default.h5'
write_path = 'predict'

# for each slice estract patches and stack
def create_slice_testing(img_dir_name):
    # empty matrix to hold patches
    patches_training_imgs_3d = np.empty(shape=[0, patch_size, patch_size, patch_size], dtype='int16')
    patches_training_gtruth_3d = np.empty(shape=[0, patch_size, patch_size, patch_size, num_classes], dtype='int16')
    images_train_dir = os.listdir(train_imgs_path)
    j = 0

	# volume 
    img_name = img_dir_name + '_hist.nii.gz'
    img_name = os.path.join(train_imgs_path, img_dir_name, img_name)
    print(img_name)

	# mask 
    img_mask_name = img_dir_name + '_mask.nii.gz'
    img_mask_name = os.path.join(train_imgs_path, img_dir_name, img_mask_name)

	# load volume and mask
    img = nib.load(img_name)
    img_data = img.get_data()
    img_data = np.squeeze(img_data)

    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_data()
    img_mask_data = np.squeeze(img_mask_data)

    patches_training_imgs_3d_temp = np.empty(shape=[0, patch_size, patch_size, patch_size], dtype='int16')

    # extract patches of the jth volume image
    imgs_patches, rows, cols, depths = extract_3d_patches_one_slice(img_data, img_mask_data)

    # update database
    patches_training_imgs_3d_temp = np.append(patches_training_imgs_3d_temp, imgs_patches, axis=0)

    patches_training_imgs_3d = np.append(patches_training_imgs_3d, patches_training_imgs_3d_temp, axis=0)

    j += 1

    # convert to single precision
    patches_training_imgs_3d = patches_training_imgs_3d.astype('float32')
    patches_training_imgs_3d = np.expand_dims(patches_training_imgs_3d, axis=4)
    return patches_training_imgs_3d, rows, cols, depths


# extract 3D patches
def extract_3d_patches_one_slice(img_data, mask_data):
    patch_shape = (patch_size, patch_size, patch_size)

    # empty matrix to hold patches
    imgs_patches_per_volume = np.empty(shape=[0, patch_size, patch_size, patch_size], dtype='int16')
    mask_patches_per_slice = np.empty(shape=[0, patch_size, patch_size, patch_size], dtype='int16')
    STEP = patch_size-1
    img_patches = extract_patches(img_data, patch_shape, extraction_reconstruct_step)
    mask_patches = extract_patches(mask_data, patch_shape, extraction_reconstruct_step)
    
    Sum = np.sum(mask_patches, axis=(3, 4, 5))
    rows, cols, depths = np.nonzero(Sum)
    N = len(rows)
    # select non-zero patches index
    selected_img_patches = img_patches[rows, cols, depths, :, :, :]

    # update database
    imgs_patches_per_volume = np.append(imgs_patches_per_volume, selected_img_patches, axis=0)
    return imgs_patches_per_volume, rows, cols, depths

# write predicted label to the final result
def write_predict(imgs_valid_predict, rows, cols, depths):
    label_predicted = np.zeros((image_rows, image_cols, image_depth, num_classes))
    label_predicted_filled = label_predicted
    label_final = np.zeros((image_rows, image_cols, image_depth))

    for index in range(0, len(rows)):
        print ('Processing patch: ', index + 1, '/', len(rows))
        row = rows[index]; col = cols[index]; dep = depths[index]
        start_row = row * extraction_reconstruct_step
        start_col = col * extraction_reconstruct_step
        start_dep = dep * extraction_reconstruct_step
        patch_volume = imgs_valid_predict[index,:,:,:,:]
        for i in range (0,patch_size):
            for j in range(0, patch_size):
                for k in range(0, patch_size):
                    prob_class0_new = patch_volume[i][j][k][0]
                    prob_class1_new = patch_volume[i][j][k][1]
                    prob_class2_new = patch_volume[i][j][k][2]
                    prob_class3_new = patch_volume[i][j][k][3]
                    
                    label_predicted_filled[start_row + i][start_col + j][start_dep + k][0] = prob_class0_new
                    label_predicted_filled[start_row + i][start_col + j][start_dep + k][1] = prob_class1_new
                    label_predicted_filled[start_row + i][start_col + j][start_dep + k][2] = prob_class2_new
                    label_predicted_filled[start_row + i][start_col + j][start_dep + k][3] = prob_class3_new                    

    for i in range(0, 256):
        for j in range(0, 128):
            for k in range(0, 256):
                prob_class0 = label_predicted_filled[i][j][k][0]
                prob_class1 = label_predicted_filled[i][j][k][1]
                prob_class2 = label_predicted_filled[i][j][k][2]
                prob_class3 = label_predicted_filled[i][j][k][3]

                prob_max = max(prob_class0, prob_class1, prob_class2, prob_class3)
                if prob_class0 == prob_max:
                    label_final[i][j][k] = 0
                elif prob_class1 == prob_max:
                    label_final[i][j][k] = 1
                elif prob_class2 == prob_max:
                    label_final[i][j][k] = 2
                else:
                    label_final[i][j][k] = 3

    return label_final

# predict function
def predict(img_dir_name):
   # create a model
    model = unet_model_3d(input_shape=config["input_shape"],
                                depth=config["depth"],
                                pool_size=config["pool_size"],
                                n_labels=config["n_labels"],
                                initial_learning_rate=config["initial_learning_rate"],
                                deconvolution=config["deconvolution"])

    model.summary()  
    
    checkpoint_filepath = 'outputs/' + checkpoint_filename
    model.load_weights(checkpoint_filepath)
    
    SegmentedVolume = np.zeros((image_rows,image_cols,image_depth))
    
    
    img_mask_name = img_dir_name + '_mask.nii.gz'
    img_mask_name = os.path.join(train_imgs_path, img_dir_name, img_mask_name)

    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_data()

    patches_training_imgs_3d, rows, cols, depths = create_slice_testing(img_dir_name)
    imgs_valid_predict = model.predict(patches_training_imgs_3d)
    label_final = write_predict(imgs_valid_predict, rows, cols, depths)

    for i in range(0, SegmentedVolume.shape[0]):
        for j in range(0, SegmentedVolume.shape[1]):
            for k in range(0, SegmentedVolume.shape[2]):
                if img_mask_data.item((i, j, k)) == 1:
                    SegmentedVolume.itemset((i,j,k), label_final.item((i, j,k)))
                else:
                    label_final.itemset((i, j,k), 0)

    print ('done')



    data = SegmentedVolume
    img = nib.Nifti1Image(data, np.eye(4))
    if num_classes == 3:
        img_name = img_dir_name + '_predicted_3class_unet3d.nii.gz'
    else:
        img_name = img_dir_name + '_predicted_4class_unet3d_extract12_hist15.nii.gz'
    nib.save(img, os.path.join('../data_new', write_path, img_name))
    print('-' * 30)

	# main
if __name__ == '__main__':
    # folder to hold outputs
    if 'outputs' not in os.listdir(os.curdir):
        os.mkdir('outputs')  
        
    images_train_dir = os.listdir(train_imgs_path)

    j=0
    
    for img_dir_name in images_train_dir:
        j=j+1
        if j:
            start_time = time.time()
   
            print('*'*50)
            print('Segmenting: volume {0} / {1} volume images'.format(j, len(images_train_dir)))
            # print('-'*30)
            if num_classes == 3:
                img_name = img_dir_name + '_predicted_3class_unet3d.nii.gz'
            else:
                img_name = img_dir_name + '_predicted_4class_unet3d_extract12_hist15.nii.gz'
            print ('Path: ', os.path.join('../data_new', write_path, img_name))
            print('*' * 50)
            predict(img_dir_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print ('Elapsed time is: ', elapsed_time)