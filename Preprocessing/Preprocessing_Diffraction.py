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
#import SRTool
import time


#"""
#Dataset informations
#Images from ANHIR-GrandChallenge
#Specimen: Human breast cell, kidney cell
#Scanner: Leica Biosystems Aperio AT2 40x
#Resolution: 0.2528 um/pixel
#This diffractive defocus model is targetted to common light field microscopy. 
#Thus, resizing does not requires. The image resolution is vasid as is.
#"""
"""
Dataset informations
Images from PatchCamelyon (PCam) which is derived from Camelyon 16 Challenge [1]
Specimen: Human Sentinel Lympnode
Scanner: NanoZoomer-XR Digital slide scanner C12000-01; Hamammatsu Photonics
Resolution: 0.226 um/pixel
This diffractive defocus model is targetted to common light field microscopy. 
Thus, resizing does not requires. The image resolution is vasid as is.

PCam dataset undersamples at x10 (96x96 pixels)
We resized as 100x100 pixels ((0.226 x 10) / (96 x 100)) um / pixels
Image was resized as 2.354 um/pixel 

[1] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning 
Algorithms for Detection of Lymph Node Metastases in Women With Breast 
Cancer. JAMA: The Journal of the American Medical Association, 318(22), 
2199–2210. doi:jama.2017.14585
"""

base_path = 'D:\\Exp_Data\\PAM_AI\\H&E_dataset\\data_diffraction'

#%% 
"""
"Three-Color" filters traditionally used on the film camera. 
Wratten #25 Red - Wratten #58 Green - Wratten #47 Blue
The Wratten numbers are come from the catalog of the master filter maker 
Frederick Wratten of the London firm of Wratten & Wainwright purchased by 
Kodak many years ago.
"""
Z_tot = np.arange(0, 5)*33.2e-6  #Light Propagation Distance
lambda_b = 425e-9 #Wratten #47
lambda_g = 525e-9 #Wratten #58
lambda_r = 700e-9 #Wratten #25
pixel_size = 2.354e-6
imsize = 100; Nx = imsize; Ny = imsize;
xx, yy =np.meshgrid(np.arange(0, imsize), np.arange(0, imsize))
Xi0 = 1 / (pixel_size * imsize); Eta0 = 1 / (pixel_size * imsize)
Xi_max = 1/pixel_size; Eta_max = 1/pixel_size

Xi = np.concatenate((np.arange(0,int(imsize/2))*Xi0, (np.arange(0,int(imsize/2))*Xi0)-  Xi_max/2), axis = 0).reshape(1,imsize)
Eta = np.concatenate((np.arange(0,int(imsize/2))*Eta0, (np.arange(0,int(imsize/2))*Eta0)-  Eta_max/2), axis = 0).reshape(imsize,1)


#%% image clipping to 100 x 100 to generate Ground Truth Image
rawdir = base_path + '\\' + '002_GroundTruth_part'
savedir = base_path + '\\' + '003_Diffracted_data'
if not(os.path.exists(savedir)):
    os.mkdir(savedir)
filelist = [f for f in glob.glob(rawdir + "**/*.png", recursive=True)]
img_size = 100



for Z in Z_tot:
    prop_dist =  ('%03d' % (Z*1e6)) + 'um'
#    imgsavdir = savedir + '\\' + prop_dist 
    imgsavdir = savedir + '\\tot'
            
    if not(os.path.exists(imgsavdir)):
        os.mkdir(imgsavdir)
    # calculation of propagation phase with regard to each wavelength (color)
    UU_r = np.exp(1j*np.pi*lambda_r*Z*Eta**2) * np.exp(1j*np.pi*lambda_r*Z*Xi**2)
    UU_g = np.exp(1j*np.pi*lambda_g*Z*Eta**2) * np.exp(1j*np.pi*lambda_g*Z*Xi**2)
    UU_b = np.exp(1j*np.pi*lambda_b*Z*Eta**2) * np.exp(1j*np.pi*lambda_b*Z*Xi**2)

    for file in filelist:
        img = cv.imread(file, cv.IMREAD_COLOR)     # image read
#        IMG_B = np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(img[:,:,0]))))
#        IMG_B = np.fft.fftshift(np.fft.fft2(img[:,:,0]))
#        IMG_G = np.fft.fftshift(np.fft.fft2(img[:,:,1]))
#        IMG_R = np.fft.fftshift(np.fft.fft2(img[:,:,2]))
        IMG_B = (np.fft.fft2(img[:,:,0]))
        IMG_G = (np.fft.fft2(img[:,:,1]))
        IMG_R = (np.fft.fft2(img[:,:,2]))
        img_B_P = np.abs(np.fft.ifft2(IMG_B * UU_r))
        Tx, Ty = np.where(img_B_P>255); img_B_P[Tx,Ty] = 255;
        Tx, Ty = np.where(img_B_P<0); img_B_P[Tx,Ty] = 0;
        img_G_P = np.abs(np.fft.ifft2(IMG_G * UU_r))
        Tx, Ty = np.where(img_G_P>255); img_G_P[Tx,Ty] = 255;
        Tx, Ty = np.where(img_G_P<0); img_G_P[Tx,Ty] = 0;
        img_R_P = np.abs(np.fft.ifft2(IMG_R * UU_r))
        Tx, Ty = np.where(img_R_P>255); img_R_P[Tx,Ty] = 255;
        Tx, Ty = np.where(img_R_P<0); img_R_P[Tx,Ty] = 0;
        img_P = np.zeros([imsize,imsize,3])
        img_P[:,:,0] = img_B_P
        img_P[:,:,1] = img_G_P
        img_P[:,:,2] = img_R_P
        name = os.path.basename(file)
        name = os.path.splitext(name)[0]
        cv.imwrite(imgsavdir + '\\' + name + '_' + prop_dist + '.png', img_P)
        #img_P_n = img_P / np.max(img_P)
        #plt.subplot(1,2,1); plt.imshow(img)
        #plt.subplot(1,2,2); plt.imshow(img_P_n)
         
#$$ Diffraction calculation , saving
#note that the color order of CV is (BGR)
#$$ Diffraction image clipping
#$$ ref Code     
#filename = rawdir + '\\' + '00a25e8f04848fe149f609fc10b891fdf67080d7.png'
#img = cv.imread(filename,cv.IMREAD_COLOR)
##     IMG_B = np.fft.fftshift(np.fft.fft2(img[:,:,0]))
##     IMG_G = np.fft.fftshift(np.fft.fft2(img[:,:,1]))
##     IMG_R = np.fft.fftshift(np.fft.fft2(img[:,:,2]))
#IMG_B = (np.fft.fft2(img[:,:,0]))
#IMG_G = (np.fft.fft2(img[:,:,1]))
#IMG_R = (np.fft.fft2(img[:,:,2]))
#img_B_P = np.abs(np.fft.ifft2(IMG_B * UU_r))
#Tx, Ty = np.where(img_B_P>255); img_B_P[Tx,Ty] = 255;
#Tx, Ty = np.where(img_B_P<0); img_B_P[Tx,Ty] = 0;
#img_G_P = np.abs(np.fft.ifft2(IMG_G * UU_r))
#Tx, Ty = np.where(img_G_P>255); img_G_P[Tx,Ty] = 255;
#Tx, Ty = np.where(img_G_P<0); img_G_P[Tx,Ty] = 0;
#img_R_P = np.abs(np.fft.ifft2(IMG_R * UU_r))
#Tx, Ty = np.where(img_R_P>255); img_R_P[Tx,Ty] = 255;
#Tx, Ty = np.where(img_R_P<0); img_R_P[Tx,Ty] = 0;
#img_P = np.zeros([imsize,imsize,3])
#img_P[:,:,0] = img_B_P
#img_P[:,:,1] = img_G_P
#img_P[:,:,2] = img_R_P
##img_P_n = img_P / np.max(img_P)
#plt.subplot(1,2,1); plt.imshow(img)
#plt.subplot(1,2,2); plt.imshow(img_P_n)