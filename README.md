# Brain-Tissue-segmentation
Segmentation of brain tissues in MRI image has a number of applications in diagnosis, surgical
planning, and treatment of brain abnormalities. However, it is a time-consuming task to be performed
by medical experts. In addition to that, it is challenging due to intensity overlap between the different
tissues caused by the intensity homogeneity and artifacts inherent toMRI. Tominimize this effect, it
was proposed to apply histogram based preprocessing. The goal of this project is to develop a robust
and automatic segmentation of WhiteMatter (WM) and GrayMatter (GM)) and Cerebrospinal Fluid
(CSF) of the human brain.
To tackle the problem, we have proposed Convolutional Neural Network (CNN) based approach and
probabilistic Atlas. U-net  is one of the most commonly used and best-performing architecture
in medical image segmentation, and we have used both 2D and 3D versions. The performance was
evaluated using Dice Coefficient (DSC), Hausdorff Distance (HD) and Average Volumetric Difference
(AVD).


### Libraries
The code has been tested with the following configuration

- h5py == 2.7.0
- keras == 2.0.2
- nibabel == 2.1.0
- nipype == 0.12.1
- python == 2.7.12
- scipy == 0.19.0
- sckit-image == 0.13.0
- sckit-learn == 0.18.1
- tensorflow == 1.0.1
- tensorflow-gpu == 1.0.1


## How to run it
Once all the libraries above have been installed, the following step is to run the jupyter notebook on the folder containing the iSeg2017.ipynb file. 
