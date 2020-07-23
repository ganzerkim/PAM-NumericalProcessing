# Photo-acoustic-Microscopy-Image-Refocusing

- Method and apparatus for numerical refocusing of photoacoustic microscopy based on super-resolution convolutional neural network

## PAM Refocusing (Gaussian PSF Model)

In this section, PAM refocusing code and code execution sequene is briefly explained.

### Code Execution Sequence

#### Data prepration

'Preprocessing_Gaussian.py' at folder 'Preprocessing'

1. kmeans clustering based on image color.
   1. Image saved in a subfolder named **'001_cluster'** from the base folder
2. Image resized as 100 x 100 to generating Ground Truth image.
   1. Image saved in a subfolder named **'002_GroundTruth'** from the base folder
3. Gaussian blurring with  σ = 0.5, 1, 1.5, 2, image size with 100 x 100.
   1. Image saved in a subfolder named **'003_Classification_data'** from the base folder
4. Image resized as 50 x 50 to generating SR training data
   1. Image saved in a subfolder named **'004_SRdata'** from the base folder

#### Classification Gaussian sigma

'classification_gaussian_sigma.py' at folder 'Classification_Gaussian_sigma'

#### Numerical Refocusing

'refocus_Train.py' at folder 'Numerical_Refocusing'

## Microscopy Refocusing (Diffractive PSF Model)

In this section, code and code execution sequene of Microscopy refocusing which follows diffractive PSF model is briefly explained.

This code is written for providing a simulation result of the paper "Deep learning super resolution based Numerical refocusing in histology image"

Note that this code uses dataset from ANHIR Grand Challenge 'https://anhir.grand-challenge.org/Intro/'

Borovec, J., Kybic, J., et al. (2020). **[ANHIR: Automatic Non-rigid Histological Image Registration Challenge](https://www.researchgate.net/publication/340497388_ANHIR_Automatic_Non-rigid_Histological_Image_Registration_Challenge)**. IEEE Transactions on Medical Imaging (TMI). doi: [10.1109/TMI.2020.2986331](https://doi.org/10.1109/TMI.2020.2986331)

#### Data Prepration

'Preprocessing_Diffraction.py' at folder 'Preprocessing'

