#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:07:41 2017

@author: Vu Hoang Minh
"""

# image shape
VOLUME_ROWS = 256
VOLUME_COLS = 128
VOLUME_DEPS = 256

# number of classes
NUM_CLASSES = 4

# patch extract
PATCH_SIZE = 32

if PATCH_SIZE==64:
    EXTRACTTION_STEP = 20
    EXTRACTTION_STEP_CSF = 5
elif PATCH_SIZE==32:
    EXTRACTTION_STEP = 10
    EXTRACTTION_STEP_CSF = 5
elif PATCH_SIZE==16:
    EXTRACTTION_STEP = 9
    EXTRACTTION_STEP_CSF = 4
    
# training configs
UNET_MODEL = 0
if UNET_MODEL==0:
    MODEL = 'default'    
elif UNET_MODEL==1:
    MODEL = 'reduced'
elif UNET_MODEL==2:
    MODEL = 'extended'
elif UNET_MODEL==3:
    MODEL = 'extended2'    
    
BASE = PATCH_SIZE
SMOOTH = 1.
NUM_EPOCHS  = 500
BATCH_SIZE  = 16
DEPTH = 5
EXTRACTION_RECONSTRUCT_STEP = 32


if PATCH_SIZE==64:
    PATIENCE = 10  
elif PATCH_SIZE==32:
    PATIENCE = 20  
elif PATCH_SIZE==16:
    PATIENCE = 10

# output
IMAGE_TYPE = '3d_whole_image'    