# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:44:22 2020

@author: MinGeon Kim
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
Image.MAX_IMAGE_PIXELS = None # to avoid image size warning


#%%
# path configuration / initialization
base_path = 'D:\\Exp_Data\\PAM_AI\\H&E_dataset\\data\\'
disp_cluster = False

#%%
"""
K-mean clustering based on image color value
"""
path_img_raw = base_path + '000_raw\\'
#(Clustering) initialize data structure
imgs_dict = SRTool.get_files(path_img_raw, size = (224, 224))
#create Keras NN model.
model = SRTool.get_model()
#Feed images through the model and extract feature vectors
img_feature_vector = SRTool.feature_vectors(imgs_dict, model)
images = list(img_feature_vector.values())
##(Clustrering) Elbow method to find Optimal cluseter number K
#fns = list(img_feature_vector.keys())
#sum_of_squared_distances = []
#K = range(1, 30)
#for k in K:
#    km = KMeans(n_clusters = k)
#    km = km.fit(images)
#    sum_of_squared_distances.append(km.inertia_)
#plt.plot(K, sum_of_squared_distances, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Sum_of_squared_distances')
#plt.title('Elbow Method For Optimal cluster number k')
#plt.show()
## note that the optimal cluster number was 4,
## then this section is not required to proceed for further procedure.
## We set number of cluster as 'n_cluster = 4' in next section.

#(Clustrering) K means clustering based on image feature
cluster_path = base_path + '001_cluster\\'
if not(os.path.exists(cluster_path)):
    os.mkdir(cluster_path)
n_clusters = 7
kmeans = KMeans(n_clusters, init = 'k-means++')
kmeans.fit(images)
y_kmeans = kmeans.predict(images)
file_names = list(imgs_dict.keys())
#(Clustrering) Clssified image saving

for c in range(0, n_clusters):
    if not os.path.exists(cluster_path + 'cluster_' + str(c)):
        os.mkdir(cluster_path + 'cluster_' + str(c))
        
for fn, cluster in zip(file_names, y_kmeans):
    image = cv.imread(path_img_raw + fn)
    cv.imwrite(cluster_path + 'cluster_' + str(cluster) + '\\' + fn, image)

#(Clustrering) image display from cluster_n
if disp_cluster == True:
    cluster_n = 4
    fig = plt.figure(figsize = (14, 14))
    cluster_disp_path = cluster_path + 'cluster_' + str(cluster_n) + '\\'
    images = [file for file in os.listdir(cluster_disp_path)]
    for cnt, data in enumerate(images[1:30]):
        y = fig.add_subplot(6, 5, cnt + 1)
        img = mpimg.imread(cluster_disp_path + data)
        y.imshow(img)
        plt.title('cluster' + str(cluster_n))
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        
#%%
"""
Image resizing to 100 x 100. Generating groundtruth image applied to training.
"""
cluster_n = 4
imgdir = cluster_path + 'cluster_' + str(cluster_n) 
savedir =  base_path + '002_GroundTruth\\'
if not(os.path.exists(savedir)):
    os.mkdir(savedir)
filelist = [f for f in glob.glob(imgdir + "**/*.tif", recursive=True)]
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

#%%
"""
Generating gaussian blurring image, size of 100 x 100, 
for generating blurring classification training and test dataset.
"""
sigma_dict = [0.5, 1, 1.5, 2]
imgdir = base_path + '002_GroundTruth\\'
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
if not(os.path.exists(base_path + '003_Classification_data')):
    os.mkdir(base_path + '003_Classification_data')
for sigma in sigma_dict:
    savedir = base_path + '003_Classification_data\\sigma_' + str(float(sigma))
    if not(os.path.exists(savedir)):
        os.mkdir(savedir)
    hsize = int(np.ceil(sigma * 6))
    if hsize % 2 == 0:
        hsize += 1
    frame_num = 1    
    for file in filelist:
        img = cv.imread(file)
        img = np.array(img)
        blurred_img = cv.GaussianBlur(img, (hsize, hsize), sigma)
        name = os.path.basename(file)
        name = os.path.splitext(name)[0]    
        cv.imwrite(savedir + '\\' + name + '_sigma' + str(float(sigma)) + '.png', blurred_img)    
        print(frame_num)
        frame_num += 1
#%%
"""
Generating downscaled, worst blurred (sigma=2) image
"""
imgdir = base_path + '003_classification_data\\sigma_' + str(float(sigma_dict[-1]))
savedir =  base_path + '004_SRDownscaledData\\'
if not(os.path.exists(savedir)):
    os.mkdir(savedir)
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
frame_num = 1
img_size = 50
for file in filelist:
    img = cv.imread(file, cv.IMREAD_COLOR)
    resize_image = cv.resize(img, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]
    cv.imwrite(savedir + '\\' + name[0:-9] + '.png', resize_image)
    print(frame_num)
    frame_num += 1
#%%
"""
Packing Gaussian Sigma Training / Test dataset
"""    
# Specifying image size
image_w = 100
image_h = 100
categories = ["ori","sigma_0.5","sigma_1.0", "sigma_1.5", "sigma_2.0"]
nb_classes = len(categories)
pixels = image_w * image_h * 3
img_path ='D:\\Exp_Data\\PAM_AI\\H&E_dataset\\data\\003_Classification_data'
# Loaging images
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    if idx == 0 :
        image_dir = base_path + '002_GroundTruth'
    else :
        image_dir = img_path + "\\" + cat
    files = glob.glob(image_dir + "\\*.png")
    for i, f in enumerate(files):
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        data = cv.normalize(data.astype(np.float64), None, 0, 1, cv.NORM_MINMAX)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

# Discrimination between the Training dataset and the Test dataset
# X: image, Y: label, M_test: Test data, M_train: Training data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
xy = (X_train, X_test, y_train, y_test)
print('>>> data saving ...')
np.save(base_path + 'sigma_classification_trainingdata_' + time.strftime('%y%m%d')+ '.npy', xy, allow_pickle=True)
print("ok,", len(Y))
    