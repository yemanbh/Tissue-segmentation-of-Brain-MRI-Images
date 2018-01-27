# image shape
VOLUME_ROWS = 256
VOLUME_COLS = 128
VOLUME_DEPS = 256

# number of classes
NUM_CLASSES = 4

# patch extract
PATCH_SIZE = 64

if PATCH_SIZE==64:
    EXTRACTTION_STEP = 12
    EXTRACTTION_STEP_CSF = 5
elif PATCH_SIZE==32:
    EXTRACTTION_STEP = 4
    EXTRACTTION_STEP_CSF = 4
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
    
BASE = PATCH_SIZE
SMOOTH = 1.
NUM_EPOCHS  = 500
BATCH_SIZE  = 64

extraction_reconstruct_step = PATCH_SIZE

if PATCH_SIZE==64:
    PATIENCE = 10  
elif PATCH_SIZE==32:
    PATIENCE = 20  
elif PATCH_SIZE==16:
    PATIENCE = 10

# output
IMAGE_TYPE = '2d_whole_image'    
