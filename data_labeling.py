# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:41:58 2019

@author: MG
"""
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

train = pd.read_csv("C:\\Users\\MG\\Desktop\\H&E New dataset\\train_labels.csv", sep=",", dtype = 'unicode')
train.head()
train.columns

train_image = []

for i in tqdm(range(train.shape[0])):
    img = image.load_img('Multi_Label_dataset/Images/'+train['Id'][i]+'.png',target_size=(100,100,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

X = np.array(train_image)
X.shape
plt.imshow(X[2])

train['label'][2]
y = np.array(train.drop(['label'],axis=1))
y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

dataset_path = os.path.join(base_path, 'dataset')
if not(os.path.exists(dataset_path)):
    os.mkdir(dataset_path)
    
np.save('D:\\Data set\\H&E data\\sample_dataset\\' + 'X_train.npy', np.array(X_train))
np.save('D:\\Data set\\H&E data\\sample_dataset\\' + 'y_train.npy', np.array(y_train))
np.save('D:\\Data set\\H&E data\\sample_dataset\\' + 'X_val.npy', np.array(X_test))
np.save('D:\\Data set\\H&E data\\sample_dataset\\' + 'y_val.npy', np.array(y_test))

print(X_train[-1].shape, y_train[-1].shape)
print(X_test[-1].shape, y_test[-1].shape)