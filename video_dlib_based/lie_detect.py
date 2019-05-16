# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:51:52 2019

@author: Muhammad Shahbaz (MS-18-IT-507815)
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from face_features import extract_face_features
from joblib import load
from sys import argv
import argparse

parser = argparse.ArgumentParser(description='a lie detector program, which can identify if the subject in a video is lying or not')
parser.add_argument('--video-path', help='the path of the video to test')
parser.add_argument('--show-video', action='store_true', default=True, help='shows video feature extraction process for debug purposes')
args = parser.parse_args()

X = extract_face_features(args.video_path, show_video=args.show_video, image_res_to_use=800)
if X and any(X):
    X = np.array(X)
    std_scaler = load('out/standard_scaler.dump')
    X_std = std_scaler.transform(X)
    pca = load('out/pca.dump')
    X_std_pca = pca.transform(X_std)
    min_max_scaler = load('out/min_max_scaler.joblib')
    X_std_pca_normal = min_max_scaler.transform(X_std_pca)

    clf = load('out/clf.joblib')
    Y = clf.predict(X_std_pca_normal)
    prediction = 'lying' if Y[0] == 1 else 'not lying'
    print('prediction: subject is %s' % prediction)
else:
    print('-- invalid path/video quality/length, feature extraction failed.')
