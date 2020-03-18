# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:25:23 2020

@author: MG
"""

import cv2 as cv
import os
import numpy as np
from keras.models import load_model, Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import matplotlib.image as mpimg


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

imgs_dict = get_files(path_to_files = 'C:\\Users\\MG\\Desktop\\dataset\\patchdata\\set2\\', size = (224, 224))

#create Keras NN model.
model = get_model()

#Feed images through the model and extract feature vectors
img_feature_vector = feature_vectors(imgs_dict, model)

#%%
#Elbow method to find Optimal K

images = list(img_feature_vector.values())
fns = list(img_feature_vector.keys())
sum_of_squared_distances = []

K = range(1, 30)
for k in K:
    km = KMeans(n_clusters = k)
    km = km.fit(images)
    sum_of_squared_distances.append(km.inertia_)
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


#%%
kmeans = KMeans(n_clusters = 4, init = 'k-means++')
kmeans.fit(images)
y_kmeans = kmeans.predict(images)
file_names = list(imgs_dict.keys())

n_clusters= 4
cluster_path = 'C:\\Users\\MG\\Desktop\\dataset\\patchdata\\'
path_to_files = 'C:\\Users\\MG\\Desktop\\dataset\\patchdata\\set2\\'

for c in range(0, n_clusters):
    if not os.path.exists(cluster_path + 'cluster_' + str(c)):
        os.mkdir(cluster_path + 'cluster_' + str(c))
        
for fn, cluster in zip(file_names, y_kmeans):
    image = cv.imread(path_to_files + fn)
    cv.imwrite(cluster_path + 'cluster_' + str(cluster) + '\\' + fn, image)

fig = plt.figure(figsize = (14, 14))

cluster_path = 'C:\\Users\\MG\\Desktop\\dataset\\patchdata\\cluster_3\\'
images = [file for file in os.listdir(cluster_path)]

for cnt, data in enumerate(images[1:30]):
    
    y = fig.add_subplot(6, 5, cnt + 1)
    img = mpimg.imread(cluster_path + data)
    y.imshow(img)
    plt.title('cluster_3')
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
