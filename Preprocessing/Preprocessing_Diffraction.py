# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:54:01 2020

@author: JGSHIN
"""
import cv2 as cv
import os
import glob
import numpy as np
from PIL import Image
from keras.models import load_model, Model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import matplotlib.image as mpimg
import SRTool
import time

base_path = 'D:\\Exp_Data\\PAM_AI\\ANHIR_dataset\\data'

"""
Dataset informations
Images from ANHIR-GrandChallenge
Specimen: Human breast cell, kidney cell
Scanner: Leica Biosystems Aperio AT2 40x
Resolution: 0.2528 um/pixel
This diffractive defocus model is targetted to common light field microscopy. 
Thus, resizing does not requires. The image resolution is vasid as is.
"""


#%% image clipping to 100 x 100 to generate Ground Truth Image
rawdir = base_path + '\\' + '000_raw'
savedir = base_path + '\\' + '001_clip_GT'
if not(os.path.exists(savedir)):
    os.mkdir(savedir)
filelist = [f for f in glob.glob(rawdir + "**/*.jpg", recursive=True)]
img_size = 100
for file in filelist:
     img = cv.imread(file, cv.IMREAD_COLOR)


    
    
#$$ Diffraction calculation , saving
#note that the color order of CV is (BGR)

    
#$$ Diffraction image clipping
     