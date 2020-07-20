# Photo-acoustic-Microscopy-Image-Refocusing
- Method and apparatus for numerical refocusing of photoacoustic microscopy based on super-resolution convolutional neural network

## PAM Refocusing (Gaussian PSF Model)

In this section, PAM refocusing code and code execution sequene is briefly explained.

### Code Execution Sequence

#### Data prepration

1. image_kmeans_clustering.py: Classify similar images. Her, it is applied to to find images of similar biological tissues.
2. Dataset_maker.py: Image clipping, Gaussian blurring
3. data_labeling.py:: data labeling

#### Classification Gaussian Sigma 

1. classification_gaussian_sigma.py: Training and execution for image blurring classification deep neural network

![](https://github.com/ganzerkim/PAM-NumericalProcessing/blob/master/Classification_Gaussian_sigma/train_result_img/Classification_of_the_gaussian_sigma_results.png?raw=true)

#### Numerical_Refocusing

1. refocus_Train.py: Training and execution for numerical image refocusing

#### ETC.

1. EDA&Data_balancing.py: Data balancing, Matching number of images between cancer and non-cancer image to train deep-neural network for cancer diagnosis.



## Microscopy Refocusing (Diffractive PSF Model)

In this section, code and code execution sequene of Microscopy refocusing which follows diffractive PSF model is briefly explained.

### 