# -*- coding: utf-8 -*-
"""
Created on Mon May 13 04:38:57 2019

@author: Muhammad Shahbaz (MS-18-IT-507815)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from joblib import load, dump
from statistics import mean, stdev

def apply_pca(X, option, n_components = 26):
    scaler = None
    pca = None

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    scaler.fit(X)
    standardized_data = scaler.transform(X)
    pca.fit(standardized_data)
    principal_components = pca.transform(standardized_data)

    if option == 1:
        dump(scaler, 'out/full_length_data_standard_scaler.dump')
        dump(pca, 'out/full_length_data_pca.dump')
        dump(principal_components, 'out/full_length_data_principal_components.dump')
    else:
        dump(scaler, 'out/clip_based_data_standard_scaler.dump')
        dump(pca, 'out/clip_based_data_pca.dump')
        dump(principal_components, 'out/clip_based_data_principal_components.dump')

    plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
    plt.axis([1,len(pca.explained_variance_ratio_),0,1])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.title('Scree Graph')
    plt.grid()
    plt.show()

    print(np.sum(pca.explained_variance_ratio_))

    return principal_components

if __name__ == '__main__':
    """
    option 1 => uses the full length audio training data
    option 2 => uses the individual clip (4 sec chunks of every audio) training data
    """

    option = 1

    data = None
    if option == 1:
        data = load('out/full_length_data.dump')
        X = data[:,1:-1]
        princial_components = apply_pca(X, option, n_components=26)
        data[:,1:27] = princial_components
        data[:,27] = data[:,-1]
        dump(data[:,:28], 'out/full_length_data_pca_reduced.dump')
    else:
        data = load('out/clip_based_data.dump')
        X = data[:,1:-1]
        princial_components = apply_pca(X, option, n_components=22)
        data[:,1:23] = princial_components
        data[:,23] = data[:,-1]
        dump(data[:,:24], 'out/clip_based_data_pca_reduced.dump')


