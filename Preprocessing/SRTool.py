# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:41:22 2020

@author: JGSHIN
"""

import cv2 as cv
import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model




#%%
def get_model(layer = 'fc2'):
    
    base_model = VGG16(weights = 'imagenet', include_top = True)
    model = Model(inputs = base_model.input, outputs = base_model.get_layer(layer).output)
    
    return model

"""
from google.colab import drive
drive.mount('\\content\\drive')
"""
def get_files(path_to_files, size):
    fn_imgs = []
    files = [file for file in os.listdir(path_to_files)]
    for file in files:
        img = cv.resize(cv.imread(path_to_files + file), size)
        fn_imgs.append([file, img])
    
    return dict(fn_imgs)


def feature_vector(img_arr, model):
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis = 2)
        
        
        #(1, 224, 224, 3)
    arr4d = np.expand_dims(img_arr, axis = 0)
    arr4d_pp = preprocess_input(arr4d)
        
    return model.predict(arr4d_pp)[0, :]

def feature_vectors(imgs_dict, model):
    f_vect = {}
    for fn, img in imgs_dict.items():
        f_vect[fn] = feature_vector(img, model)
    return f_vect
