# FeaUNet
The training code will be released after the paper is accepted.

## Requirements 
#### Matlab
#### Cuda

## Test
Download pretrained GMM prior from https://pan.baidu.com/s/1otlcyO3opPuZbJeifLgi-g (passwd: m03j) and place it in the correct path.

    gradGmm_gray_single_2048_06_joint_Gk2.mat  -------->  /TestCode/denoiseUtilities/FeaEPLL/
  
    gradGmm_color_single_1024_05_joint_Gk2.mat -------->  /TestCode/denoiseUtilities/FeaEPLL/

    Gmm_gray_single_2048_08_v2_Gk1.mat         -------->  /TestCode/denoiseUtilities/EPLL_2048/

Download testsets (Set12,BSD68,Urban100) from https://pan.baidu.com/s/1CtsgWUk6Y-2M8xqTrVXnmA (passwd: c9ki) and place it in the correct path.

run main_Test_Gray.m for Grayscale Denoising or main_TestColor.m for Color Denoising.
