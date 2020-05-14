# HRAN
### This repository is an official PyTorch implementation of the paper "HRAN : Hybrid Residual Attention Network for Single Image Super Resolution".

Paper can be download from <a href="https://ieeexplore.ieee.org/document/8844684">HRAN</a> 

All test datasets (Preprocessed HR images) can be downloaded from <a href="https://www.jianguoyun.com/p/DcrVSz0Q19ySBxiTs4oB">here</a>.

All original test datasets (HR images) can be downloaded from <a href="https://www.jianguoyun.com/p/DaSU0L4Q19ySBxi_qJAB">here</a>.

The trained models are available on <a href="https://drive.google.com/drive/folders/1MC3jXCxnKeJDElkFLmCvCsjmZzSGPIs0?usp=sharing"> Google Drive</a>

--------------------

## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Results](#results)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)

## Introduction
The extraction and proper utilization of convolutional neural network (CNN) features have a significant impact on the performance of image super-resolution (SR). Although CNN features contain both spatial and channel information, current deep learning techniques for SR often suffer to maximize the performance due to using either the spatial information or channel information. Moreover, they integrate such information within a deep or wide network rather than exploiting all the available features, eventually resulting in high computational complexity. To address these issues, we present a binarized feature fusion (BFF) structure that utilizes the extracted features from global residuals (GR) in an effective way. Each GR consists of multiple hybrid residual attention blocks (HRAB) that effectively integrates the multiscale feature extraction module and channel attention mechanism in a single block. Furthermore, to save computational power, instead of using a large filter size, we use convolutions with different dilation factors to extract multiscale features. We also propose to adopt global skip connections (GSC), short skip connections (SSC), long skip connections (LSC) and GR structure to ease the flow of information without losing important features details. In the paper, we call this overall network architecture as hybrid residual attention network (HRAN). In the experiment, we have observed the efficacy of our method against the state-of-the-art methods for both the quantitative and qualitative comparisons.


![HRAB](/Figures/HRAB.png)
Hybrid Residual attention block (HRAB) architecture.
![HRAN](/Figures/HRAN.png)
The architecture of our proposed hybrid residual attention network (HRAN).

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

## Results
### Quantitative Results
![PSNR_SSIM_BI](/Figures/results.gif)

Quantitative results with BI degradation model. Best, 2nd Best and 3rd Best Results are Respectively Shown With Magenta, Blue, and Green Colors

### Memory Comparisons
![PSNR_SSIM_Parameters](/Figures/mem_comparisons.png)
Comparison of memory and performance. Results are evaluated on Urban100 (×4)

### Visual Results
![Visual_PSNR_SSIM_BI](/Figures/vis_results.png)
Visual results with Bicubic (BI) degradation (4×) on on Urban100 and Manga109 datasets

For more results, please refer to our [paper](https://ieeexplore.ieee.org/abstract/document/8844684) 

## Citation
If you find this code helpful in your research, please cite the following paper.
```
@article{muqeet2019hran,
  title={HRAN: Hybrid Residual Attention Network for Single Image Super-Resolution},
  author={Muqeet, Abdul and Iqbal, Md Tauhid Bin and Bae, Sung-Ho},
  journal={IEEE Access},
  volume={7},
  pages={137020--137029},
  year={2019},
  publisher={IEEE}
}
```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We are grateful to the authors for sharing their codes of EDSR (https://github.com/thstkdgus35/EDSR-PyTorch).


