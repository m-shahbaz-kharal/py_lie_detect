# -*- coding: utf-8 -*-
"""
Created on Tue May 14 03:43:04 2019

@author: Muhammad Shahbaz (MS-18-IT-507815)
"""
from joblib import load

def get_score(X, Y, option):
    standard_scaler = None
    pca = None
    clf = None

    if option == 1:
        standard_scaler = load('out/full_length_data_standard_scaler.dump')
        pca = load('out/full_length_data_pca.dump')
        min_max_scaler = load('out/full_length_data_min_max_scaler.dump')
        clf = load('out/full_length_data_clf.dump')
    else:
        standard_scaler = load('out/clip_based_data_standard_scaler.dump')
        pca = load('out/clip_based_data_pca.dump')
        min_max_scaler = load('out/clip_based_data_min_max_scaler.dump')
        clf = load('out/clip_based_data_clf.dump')

    standardized_data = standard_scaler.transform(X)
    principal_components = pca.transform(standardized_data)
    normal_pcs = min_max_scaler.transform(principal_components)
    score = clf.score(normal_pcs, Y)
    return score

if __name__ == '__main__':
    """
    option 1 => uses the full length audio training data
    option 2 => uses the individual clip (4 sec chunks of every audio) training data
    """

    option = 1

    from mfcc_feaures_extractor import create_dataset_using_full_clips, create_dataset_using_clip_chunks

    if option == 1:
        create_dataset_using_full_clips('test_data', 'temporary_audio_storage', 'test_data')
    else:
        create_dataset_using_clip_chunks('test_data', 'temporary_audio_storage', 'test_data')

    test_data = load('out/test_data.dump')
    score = get_score(test_data[:,1:-1], test_data[:,-1], option=option)
    print('score:', score)
