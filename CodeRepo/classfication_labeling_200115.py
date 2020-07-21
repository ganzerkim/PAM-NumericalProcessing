# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:30:38 2019

@author: mit
"""

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv

# 분류 대상 카테고리 선택하기 
base_dir = "D:\\H&E_dataset\\defocuse"
categories = ["ori","sigma_0.5","sigma_1.0", "sigma_1.5", "sigma_2.0"]
nb_classes = len(categories)

# 이미지 크기 지정 
image_w = 100 
image_h = 100
pixels = image_w * image_h * 3

# 이미지 데이터 읽어 들이기 
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = base_dir + "\\" + cat
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

# 학습 전용 데이터와 테스트 전용 데이터 구분 
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)


print('>>> data 저장중 ...')
np.save("D:\\H&E_dataset\\dataset\\defocusing_6sigma_classification_200205.npy", xy)
print("ok,", len(Y))