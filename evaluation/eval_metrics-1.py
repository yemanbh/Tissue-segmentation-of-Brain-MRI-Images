
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import nibabel as nib
from skimage import io

from nipype.algorithms.metrics import Distance

import SimpleITK as sitk

from hausdorff import hausdorff


def compute_hausdorff_distance(in1, in2, label = 'all'):
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    if label == 'all':
        # Hausdorff distance
        hausdorff_distance_filter.Execute(in1, in2)
    else:
    
        in1_array  = sitk.GetArrayFromImage(in1)
        in1_array = (in1_array == label) *1 
        in1_array = in1_array.astype('uint16')  
        img1 = sitk.GetImageFromArray(in1_array)
        
        in2_array  = sitk.GetArrayFromImage(in2)
        in2_array = (in2_array == label) *1 
        in2_array = in2_array.astype('uint16')  
        img2 = sitk.GetImageFromArray(in2_array)
        # Hausdorff distance
        hausdorff_distance_filter.Execute(img1, img2)
    return hausdorff_distance_filter.GetHausdorffDistance()

def compute_dice_coefficient(in1, in2, label  = 'all'):

    if label=='all': 
        return 2 * np.sum( (in1>0) &  (in2>0) & (in1 == in2)) / (np.sum(in1 > 0) + np.sum(in2 > 0))
    else:
        return 2 * np.sum((in1 == label) & (in2 == label)) / (np.sum(in1 == label) + np.sum(in2 == label))

def compute_volumentric_difference(in1, in2, label  = 'all'):
    if label  == 'all':
#        vol_dif  = np.sum((in1 != in2) & (in1 !=0) & (in2 !=0))
        return np.sum((in1 != in2)) / ((np.sum(in1 > 0) + np.sum(in2 > 0)))

    else:
        in1  = (in1 == label) * 1
        in2  = (in2 == label) * 1
        return np.sum((in1 != in2)) / ((np.sum(in1 > 0) + np.sum(in2 > 0)))




if __name__ ==  '__main__':
    gt  = '/home/yb/my-files/datasets/MISA_dataset_res/predict_header_data/IBSR_17_seg.nii.gz'
    pred  = '//home/yb/my-files/datasets/MISA_dataset_res/Validation_Set/IBSR_17/IBSR_17_seg.nii.gz'

#    gt_nii = nib.load(gt)
#    gt_nii  = gt_nii.get_data()
#    I1 = gt_nii[:,:,100]
##    io.imshow(np.squeeze(I1,axis=-1))
#    
#    pred_nii = nib.load(pred)
#    pred_nii  = pred_nii.get_data()
##    
##    I2 = pred_nii[:,:,100]
###    io.imshow(np.squeeze(I2, axis=-1))
##    
#    dice  = compute_dice_coefficient(gt_nii, pred_nii)
#    
#    print(dice)
##    
#    voil_dif = compute_volumentric_difference(gt_nii, pred_nii)
#    print(voil_dif)
    
    im1  = sitk.ReadImage(gt,  sitk.sitkUInt16)
    im2  = sitk.ReadImage(pred,  sitk.sitkUInt16)

    
    # I1  = sitk.GetArrayFromImage(im2)
    
    # I2 = (I1 == 2) *1 
    # I2 = I2.astype('uint16')
    
#    image = sitk.Image(256, 128, 256, sitk.sitkInt16)
#    image.SetOrigin(im1.GetOrigin())
    
    # img = sitk.GetImageFromArray(I2)
    
#    file_name  = '17_2.nii.gz'
#    sitk.WriteImage(img, file_name)
    
    haus = compute_hausdorff_distance(im1, im2, label = 3)
    print(haus)




