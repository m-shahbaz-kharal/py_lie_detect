# -*- coding: utf-8 -*-
"""
Created on Thu May  2 01:54:14 2019

@author: Muhammad Shahbaz (MS-18-IT-507815)
"""
from sklearn.model_selection import cross_val_score
from face_features import extract_face_features
from sklearn.metrics import average_precision_score
import numpy as np
from joblib import load
from glob import glob

X = []
Y = []
for file in glob('test_data/*.mp4'):
    features = extract_face_features(file)
    X.append(features)
    if file.find('lie') > 0: Y.append(1)
    else: Y.append(-1)

# scaling
std_scaler = load('out/standard_scaler.dump')
pca = load('out/pca.dump')
min_max_scaler = load('out/min_max_scaler.dump')

X_std = std_scaler.transform(X)
X_std_pca = pca.transform(X_std)
X_std_pca_normal = min_max_scaler.transform(X_std_pca)

# loading classifier
clf = load('out/clf.dump')

# testing
Y_score = clf.decision_function(X_std_pca_normal)
average_precision_score = average_precision_score(Y, Y_score)
print('average precision-recall score: {0:0.2f}'.format(average_precision_score))

