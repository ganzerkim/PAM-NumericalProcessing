# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:19:50 2020

@author: MIT-DGMIF
"""

#%%%
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None # to avoid image size warning

#Class mg_dataset_maker():
#def mg_rgb2hsv:
#def mg_resizing:
#def mg_gaussianblur:
#def mg_norm:
#def mg_sobel:
#%%%
#rgbtohsv
base_path = 'D:\\H&E_dataset'    
imgdir = "D:\\H&E_dataset\\resized(100)_re"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = os.path.join(base_path, 'H_V(100)')
savedir_2 = os.path.join(base_path, 'S_V(100)')
if not(os.path.exists(savedir)):
    os.mkdir(savedir)
if not(os.path.exists(savedir_2)):
    os.mkdir(savedir_2)
    
frame_num = 1
for file in filelist:
    img = cv.imread(file, cv.IMREAD_COLOR)
    img = np.array(img)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_img)
    
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]
    
    one = np.zeros((100, 100))
    ddd = one.astype(np.uint8)
    
    hv = cv.merge([h, ddd, v])
    sv = cv.merge([ddd, s, v])
    #cv.imwrite(savedir + '\\H\\' + name + '_h.png', h)
    #cv.imwrite(savedir + '\\S\\' + name + '_s.png', s)
    cv.imwrite(savedir + '\\' + name + '.png', hv)
    cv.imwrite(savedir_2 + '\\' + name + '.png', sv)
    print(frame_num)
    frame_num += 1

#%%%
#image resizing

imgdir = "C:\\Users\\MG\\Desktop\\H&E New dataset\\SuperResolutionData\\H&E_imageset\\20200224_SRGAN_sigma2.0(200x200)\\training_result\\prd_img(200)"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "C:\\Users\\MG\\Desktop\\H&E New dataset\\SuperResolutionData\\H&E_imageset\\20200224_SRGAN_sigma2.0(200x200)\\training_result\\prd_resize_img(100)"

#start_pos = start_x, start_y = (0, 0)
#cropped_image_size = w, h = (100, 100)
frame_num = 1
img_size = 100
for file in filelist:
    img = cv.imread(file, cv.IMREAD_COLOR)
    resize_image = cv.resize(img, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]
    cv.imwrite(savedir + '\\' + name + '.png', resize_image)
    
    print(frame_num)
    frame_num += 1


imgdir = "D:\\H&E_dataset\\S_V(100)"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\S_V(40)"

#start_pos = start_x, start_y = (0, 0)
#cropped_image_size = w, h = (100, 100)
frame_num = 1
img_size = 40
for file in filelist:
    img = cv.imread(file, cv.IMREAD_COLOR)
    resize_image = cv.resize(img, dsize = (img_size, img_size), interpolation = cv.INTER_AREA)
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]
    cv.imwrite(savedir + '\\' + name + '.png', resize_image)
    print(frame_num)
    frame_num += 1
    
    
#%%    
#image cropping
imgdir = "C:\\Users\\MG\\Desktop\\dataset\\H&Edataset"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "C:\\Users\\MG\\Desktop\\dataset\\patchdata"

start_pos = start_x, start_y = (0, 0)
cropped_image_size = w, h = (100, 100)

for file in filelist:
    img = Image.open(file)
    width, height = img.size

    frame_num = 1
    for col_i in range(0, width, w):
        for row_i in range(0, height, h):
            crop = img.crop((col_i, row_i, col_i + w, row_i + h))
            name = os.path.basename(file)
            name = os.path.splitext(name)[0]
            save_to= os.path.join(savedir, name+"_{:03}.png")
            crop.save(save_to.format(frame_num))
            frame_num += 1
#%%    
#gaussian blur
#final_sig = 2.0
#ori_sig = 0.5
#sig = ((final_sig**2) - (ori_sig**2)) ** (0.5)

size = [2.0]

for sigma in size:
    imgdir = "C:\\Users\\MG\\Desktop\\test_images\\cluster_1(test)"
    # if you want file of a specific extension (.png):
    filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
    savedir = "C:\\Users\\MG\\Desktop\\test_images\\cluster_1_100(test)_sigma_" + str(float(sigma))
    
    if not(os.path.exists(savedir)):
        os.mkdir(savedir)
        
    #val = int(1 + np.ceil(sigma) * 2)
    val = int(np.ceil(sigma * 6))
    if val % 2 == 0:
        val += 1
    
    frame_num = 1    
    for file in filelist:
        img = cv.imread(file)
        img = np.array(img)
        
        blurred_img = cv.GaussianBlur(img, (val, val), sigma)
        name = os.path.basename(file)
        name = os.path.splitext(name)[0]    
        cv.imwrite(savedir + '\\' + name + '_sigma' + str(float(sigma)) + '.png', blurred_img)    
        print(frame_num)
        
        frame_num += 1
 
    
##################################
#kernel customizing
##################################
#gaussian blur

size = [1.5]

for sigma in size:
    imgdir = "C:\\Users\\MG\\Desktop\\DGMIF\\BTM\\test_sample\\sigma_0.55"
    # if you want file of a specific extension (.png):
    filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
    savedir = "C:\\Users\\MG\\Desktop\\DGMIF\\BTM\\test_sample\\sigma_0.5_sigma_" + str(float(sigma))
    
    if not(os.path.exists(savedir)):
        os.mkdir(savedir)
        
    val = int(1 + np.ceil(sigma) * 2)
    #val = 3
    
    #kernel building
    gaussian_k = cv.getGaussianKernel(5, 1.5)
    kernel_mg = 0.117 * gaussian_k
    kernel_mg = np.array([[0.1522, 01154, -0.5352, 0.1154, 0.1522]])
    
    
    frame_num = 1    
    for file in filelist:
        img = cv.imread(file)
        img = np.array(img)
        blurred_img = cv.filter2D(img, -1, kernel_mg)
        name = os.path.basename(file)
        name = os.path.splitext(name)[0]    
        cv.imwrite(savedir + '\\' + name + '_sigma' + str(float(sigma)) + '.png', blurred_img)    
        print(frame_num)
        
        frame_num += 1

#%%

#img_dir_tmp =['D:\\H&E_dataset\\defocuse\\test_sigma_0','D:\\H&E_dataset\\defocuse\\test_sigma_1','D:\\H&E_dataset\\defocuse\\test_sigma_3','D:\\H&E_dataset\\defocuse\\test_sigma_5','D:\\H&E_dataset\\defocuse\\test_sigma_7'];
img_dir_tmp = ['','','',''];
img_dir_tmp[0] = "D:\\H&E_dataset\\defocuse\\test_sigma_0"
img_dir_tmp[1] = "D:\\H&E_dataset\\defocuse\\test_sigma_1"
img_dir_tmp[2] = "D:\\H&E_dataset\\defocuse\\test_sigma_3"
img_dir_tmp[3] = "D:\\H&E_dataset\\defocuse\\test_sigma_5"
filelist_val = [[0 for col in range(10)] for row in range(4)]
filelist_tmp = [f for f in glob.glob(img_dir_tmp[0] + "**/*.png", recursive=True)]
filelist_val[0][0:10] = filelist_tmp[0:10]
filelist_tmp = [f for f in glob.glob(img_dir_tmp[1] + "**/*.png", recursive=True)]
filelist_val[1][0:10] = filelist_tmp[1:11]
filelist_tmp = [f for f in glob.glob(img_dir_tmp[2] + "**/*.png", recursive=True)]
filelist_val[2][0:10] = filelist_tmp[0:10]
filelist_tmp = [f for f in glob.glob(img_dir_tmp[3] + "**/*.png", recursive=True)]
filelist_val[3][0:10] = filelist_tmp[1:11]

savedir = "D:\\H&E_dataset\\defocuse\\test_set"

frame_num = 0;
path_num  = 0;

for file in filelist_val[:][0]:
    img1 = Image.open(filelist_val[0][frame_num])
    img2 = Image.open(filelist_val[1][frame_num])
    img3 = Image.open(filelist_val[2][frame_num])
    img4 = Image.open(filelist_val[3][frame_num])
    img_merge = np.concatenate((np.concatenate((img1,img2),axis=1), np.concatenate((img3,img4),axis=1)),axis=0);
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]         
    cv.imwrite(savedir + '\\' + name +'_merge.png', img_merge)    
    frame_num += 1  

#%%

imgdir = "D:\\H&E_dataset\\sigma_0"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\sigma_3"

sigma = 3
val = int(1 + np.ceil(sigma * 2))


frame_num = 1
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    blurred_img = cv.GaussianBlur(img, (val, val), sigma)
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]    
    cv.imwrite(savedir + '\\' + name +'_sigma' + str(int(sigma)) + '.png', blurred_img)    
    print(frame_num)
    frame_num += 1   



imgdir = "D:\\H&E_dataset\\sigma_0"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\sigma_5"

sigma = 5
val = int(1 + np.ceil(sigma * 2))


frame_num = 1
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    blurred_img = cv.GaussianBlur(img, (val, val), sigma)
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]    
    cv.imwrite(savedir + '\\' + name +'_sigma' + str(int(sigma)) + '.png', blurred_img)    
    print(frame_num)
    frame_num += 1   
    
imgdir = "D:\\H&E_dataset\\sigma_0"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\sigma_7"


sigma = 7
val = int(1 + np.ceil(sigma * 2))



frame_num = 1
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    blurred_img = cv.GaussianBlur(img, (val, val), sigma)
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]    
    cv.imwrite(savedir + '\\' + name +'_sigma' + str(int(sigma)) + '.png', blurred_img)    
    print(frame_num)
    frame_num += 1 
 
    
    
#%%
#sobel operator
imgdir = "D:\\H&E_dataset\\defocuse\\sigma_0"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\defocuse\\test"

frame_num = 1
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_img)
    sobelx = cv.Sobel(v,cv.CV_8U,1,0,ksize=3)
    sobely = cv.Sobel(v,cv.CV_8U,0,1,ksize=3)
    sobel = sobelx + sobely
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]    
    cv.imwrite(savedir + '\\' + name + '_sobelxy.png', sobel)    
    print(frame_num)
    frame_num += 1    

##################################################
imgdir = "D:\\H&E_dataset\\defocuse\\sigma_1"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\defocuse\\edge_1"

frame_num = 1
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    sobelx = cv.Sobel(img,cv.CV_8U,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_8U,0,1,ksize=3)
    sobel = sobelx + sobely
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]    
    cv.imwrite(savedir + '\\' + name + '_sobelxy.png', sobel)    
    print(frame_num)
    frame_num += 1    

####################################################
imgdir = "D:\\H&E_dataset\\defocuse\\sigma_3"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\defocuse\\edge_3"

frame_num = 1
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    sobelx = cv.Sobel(img,cv.CV_8U,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_8U,0,1,ksize=3)
    sobel = sobelx + sobely
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]    
    cv.imwrite(savedir + '\\' + name + '_sobelxy.png', sobel)    
    print(frame_num)
    frame_num += 1    


##############################################################
imgdir = "D:\\H&E_dataset\\defocuse\\sigma_5"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\defocuse\\edge_5"

frame_num = 1
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    sobelx = cv.Sobel(img,cv.CV_8U,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_8U,0,1,ksize=3)
    sobel = sobelx + sobely
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]    
    cv.imwrite(savedir + '\\' + name + '_sobelxy.png', sobel)    
    print(frame_num)
    frame_num += 1    

##########################################################
imgdir = "D:\\H&E_dataset\\defocuse\\sigma_7"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
savedir = "D:\\H&E_dataset\\defocuse\\edge_7"

frame_num = 1
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    sobelx = cv.Sobel(img,cv.CV_8U,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_8U,0,1,ksize=3)
    sobel = sobelx + sobely
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]    
    cv.imwrite(savedir + '\\' + name + '_sobelxy.png', sobel)    
    print(frame_num)
    frame_num += 1  
#%%
img_base_path = "D:\\H&E_dataset"
x_img_path = os.path.join(img_base_path + '\\H_V(40)')
y_img_path = os.path.join(img_base_path + '\\H_V(100)')

x_img = [cv.imread(x_img_path + '\\' + s, cv.IMREAD_GRAYSCALE) for s in os.listdir(x_img_path)]
y_img = [cv.imread(y_img_path + '\\' + s, cv.IMREAD_GRAYSCALE) for s in os.listdir(y_img_path)]


x_data = []
y_data = []


for indx in range(len(x_img)):
    norm_x = cv.normalize(x_img[indx].astype(np.float64), None, 0, 1, cv.NORM_MINMAX)
    x_data.append(norm_x)
    norm_y = cv.normalize(y_img[indx].astype(np.float64), None, 0, 1, cv.NORM_MINMAX)
    y_data.append(norm_y)
    print(indx)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1)
    
dataset_path = os.path.join(img_base_path, 'dataset')
if not(os.path.exists(dataset_path)):
    os.mkdir(dataset_path)
    
np.save('D:\\H&E_dataset\\dataset\\' + 'x_train_hv(40)_220025.npy', np.array(x_train))
np.save('D:\\H&E_dataset\\dataset\\' + 'y_train_hv(100)_220025.npy', np.array(y_train))
np.save('D:\\H&E_dataset\\dataset\\' + 'x_val_hv(40)_220025.npy', np.array(x_val))
np.save('D:\\H&E_dataset\\dataset\\' + 'y_val_hv(100)_220025.npy', np.array(y_val))

print(x_train[-1].shape, y_train[-1].shape)
print(x_val[-1].shape, y_val[-1].shape)
