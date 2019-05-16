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

def apply_pca(X, n_components = 136):
    scaler = None
    pca = None

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    scaler.fit(X)
    standardized_data = scaler.transform(X)
    pca.fit(standardized_data)
    principal_components = pca.transform(standardized_data)

    dump(scaler, 'out/standard_scaler.dump')
    dump(pca, 'out/pca.dump')
    dump(principal_components, 'out/principal_components.dump')

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
    data = None
    data = load('out/train_data.dump')
    X = data[:,:-1]
    princial_components = apply_pca(X, n_components=16)
    data[:,:16] = princial_components
    data[:,16] = data[:,-1]
    dump(data[:,:17], 'out/train_data_pca_reduced.dump')


