# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:34:36 2019

@author: mit
"""
# 학습
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import utils, models, layers, optimizers
import numpy as np
import os
from PIL import Image
import os, glob
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras.utils import multi_gpu_model                              ###in case of using multi GPU
# 카테고리 지정하기
"""
#학습시
categories = ["ori", "sigma_0.5", "sigma_1.0", "sigma_1.5", "sigma_2.0"]
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 100
image_h = 100
# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("D:\\H&E_dataset\\dataset\\defocusing_6sigma_classification_200205.npy")
"""
#분석시
categories = ["ori", "sigma_0.5", "sigma_1.0", "sigma_1.5", "sigma_2.0"]
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 100
image_h = 100

# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("D:\\H&E_dataset\\dataset\\defocusing_classification_200115.npy")

imgdir = "C:\\Users\\MG\\Desktop\\DGMIF\\BTM\\test_sample\\test_img"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]

test = []
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    test.append(img)
X_train = np.array(test)

#%%
'''
# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(128))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='adam',
    metrics=['accuracy'])
# 모델 확인
print(model.summary())
'''
#%% (jgshin) 네트워크 생성
import efficientnet.keras as efn
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

base_model = efn.EfficientNetB7(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = (100, 100, 3))
x = base_model.output
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
output = Dense(nb_classes, activation = 'softmax')(x)
model = Model(base_model.input, output)
#model = multi_gpu_model(model, gpus = 2)                                       ###in case of using multi GPU
model.summary()
model.compile(optimizer=optimizers.Adam(lr = 1e-5), loss = 'categorical_crossentropy',  metrics=['accuracy'])


#%%#%% (jgshin) 훈련하기/Trainig fit

# 모델 훈련하기
#model.fit(X_train, y_train, batch_size=32, nb_epoch=5)
# 학습 완료된 모델 저장
hdf5_file = "D:\\H&E_dataset\\data\\weight_defocusing_adam100_Classification_200206.hdf5"

if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    #early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-06)])
    #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=32, callbacks=[early_stopping])
    model.save_weights(hdf5_file)
    
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].set_title('loss')
    ax[0, 0].plot(history.history['loss'], 'r')
    ax[0, 1].set_title('acc')
    ax[0, 1].plot(history.history['acc'], 'b')

    ax[1, 0].set_title('val_loss')
    ax[1, 0].plot(history.history['val_loss'], 'r--')
    ax[1, 1].set_title('val_acc')
    ax[1, 1].plot(history.history['val_acc'], 'b--')
#%%
# 모델 평가하기 
score = model.evaluate(X_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc

############################
#single image 적용
############################
# 적용해볼 이미지 
#C:\Users\MDDC\Desktop\2019-02-12 Histopathology dataset\testdata
imgdir = "D:\\H&E_dataset\\data\\test_classification"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]

test = []
for file in filelist:
    img = Image.open(file)
    img = np.array(img)
    test.append(img)
X_train = np.array(test)

#%% (jgshin) 평가/예측
# 예측
#X = X_train.astype("float") / 256
norm_x = cv.normalize(X_train.astype(np.float64), None, 0, 1, cv.NORM_MINMAX)
pred = model.predict(norm_x)  
result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환

print('Prediction of Sigma Value in Gaussian Blur  : ', categories[result[0]])  



#%% (jgshin) 최종 예측
#########################################################
#최종 예측
#########################################################
from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv
#%% (jgshin) 최종 예측 prediction
# 분류 대상 카테고리 선택하기 
base_dir = "D:\\H&E_dataset\\data\\00x_test_classification"
# categories = ["ori","sigma_0.5","sigma_1.0", "sigma_1.5", "sigma_0.5_sigma_1.9365"]
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
norm_x = cv.normalize(X.astype(np.float64), None, 0, 1, cv.NORM_MINMAX)

#%% (jgshin) 최종 예측 plot array
# class_names =["ori","sigma_0.5","sigma_1.0", "sigma_1.5", "sigma_0.5_sigma_1.9365"]
class_names = ["ori", "sigma_0.5", "sigma_1.0", "sigma_1.5", "sigma_2.0"]

predictions = model.predict(X)
pred =  predictions.astype("float16")

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == np.argmax(true_label):
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[np.argmax(true_label)]), color = color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[np.argmax(true_label)].set_color('blue')
    
num_rows = 10
num_cols = 10
num_images = num_rows * num_cols
plt.figure(figsize = (2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, Y, norm_x)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, Y)
plt.show()
