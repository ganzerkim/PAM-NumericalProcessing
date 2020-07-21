# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:42:56 2020

@author: MG
"""

import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os
import seaborn as sns
import scipy.ndimage

def eda_label_counter(csv_name):
    base_path = 'C:\\Users\\MG\\Desktop\\H&E New dataset\\CancerClassificationData'
    train = pd.read_csv(base_path + '\\' + csv_name)
    train.head(12)
    train.shape   #(4045572, 2)
    newtable = train.copy()
    train.label.isnull().sum()   #---------->NaN의 갯수 합
    train.label.value_counts()
    #Visualization of data
    return sns.countplot(train.label) #csv 파일 0, 1 비율

def label_imbalance_correction(csv_name, new_csv_name):
    
    base_path = 'C:\\Users\\MG\\Desktop\\H&E New dataset\\CancerClassificationData'
    df = pd.read_csv(base_path + '\\' + csv_name)
    df.head(10)
    cancer = df[df.label == 1]
    non_cancer = df[df.label == 0]
    df1 = cancer
    df2 = non_cancer[0:len(cancer)]
    dfr = pd.concat([df1, df2], ignore_index = True)
    df_final = dfr.sample(frac = 1).reset_index(drop = True)
    df_final.head(10)
    return df_final.to_csv(base_path + '\\' + new_csv_name, mode = 'w')

csv_name = 'train_labels.csv'
new_csv_name = 'new_label.csv'
eda_label_counter(csv_name)
label_imbalance_correction(csv_name, new_csv_name)
eda_label_counter(new_csv_name)
    