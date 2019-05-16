# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:00:37 2019

@author: Muhammad Shahbaz (MS-18-IT-507815)
"""
from joblib import load
import numpy as np

# leave one person out cross-validation
def lopo_cross_validation(data):
    all_ids = np.unique(data[:,0])

    cv_iterator = []

    for i in all_ids:
        train_indices = np.where(data[:,0] != i)
        test_indices = np.where(data[:,0] == i)
        cv_iterator.append((list(train_indices[0]), list(test_indices[0])))

    return cv_iterator

""" # under dev
def person_k_fold(data, k=10):
    sorted_data = np.array(sorted(data, key=lambda item: item[-1]))

    all_ids = np.unique(sorted_data[:,0])
    total_count = len(sorted_data)

    negative_data = np.take(sorted_data, np.where(sorted_data[:,-1] == -1))
    positive_data = np.take(sorted_data, np.where(sorted_data[:,-1] == 1))

    train_count = round(k-1.0/k)*total_count
    test_count = total_count - train_count

    test_chunk_size = round(1.0/k)*total_count

    cv_iterator = []

    for i in range(k):
        test_ids = all_ids[i*:i+test_chunk_size]
        train_indices = negative_data[]
        test_indices = np.where(data[:,0] == i)
        cv_iterator.append((list(train_indices[0]), list(test_indices[0])))

    return cv_iterator

"""

if __name__ == '__main__':
    """
    option 1 => uses the full length audio training data
    option 2 => uses the individual clip (4 sec chunks of every audio) training data
    """

    option = 1

    data = None

    if option == 1:
        data = load('out/full_length_data_pca_reduced.dump')
    else:
        data = load('out/clip_based_data_pca_reduced.dump')
    result = lopo_cross_validation(data)


