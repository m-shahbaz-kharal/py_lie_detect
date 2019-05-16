# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:37:20 2019
@author: Muhammad Shahbaz (MS-18-IT-507815)
"""

from sklearn import preprocessing
from collections import Counter
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
import seaborn
from joblib import dump, load
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# a lot of warnings
import warnings
warnings.filterwarnings('ignore')

# loading data
data = load('out/train_data_pca_reduced.dump')
X = data[:,:-1]
Y = data[:,-1]

# checking if data is normalized or not by plotting ranges
number_of_points, dimensions = X.shape

x_ranges = []
for i in range(dimensions): x_ranges.append([X[:,i].min(), X[:,i].max()])
for i, xr in enumerate(x_ranges): plt.plot(xr, [i+1, i+1], 'ro-')
plt.title('Before Normalizing Attribute Values')
plt.show()

# normalizing data, it shouldn't be applied on sparsed data (as stated in ben hur's paper)
# but I'm considering that data is not sparced (as we don't know how to find sparsity of data)
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
train_data_normalized = scaler.transform(X)

# now we can see that data is normalized and sensitivity problem due to feature scale will now be minimum
number_of_points, dimensions = train_data_normalized.shape

x_ranges = []
for i in range(dimensions): x_ranges.append([train_data_normalized[:,i].min(), train_data_normalized[:,i].max()])
for i, xr in enumerate(x_ranges): plt.plot(xr, [i+1, i+1], 'ro-')
plt.title('After Normalizing Attribute Values')
plt.show()

# checking class distribution
counts = dict(Counter(Y))
print ('Counts of labels:', counts)
print ('Data is not balanced, we should use weighted SVM (here we are using balanced, which works it out automatically) and Stratified K-Fold.')

# defining classifiers to be used in grid search
knn_clf = KNeighborsClassifier()
knn_params = {'n_neighbors':[1,3,5,7,9,11,13,15]}

linear_svm_clf = LinearSVC(class_weight='balanced')
linear_svm_params = {'C':[0.00001,0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000]}

poly_svm_clf = SVC(class_weight='balanced', kernel='poly')
poly_svm_params = {'C': [0.00001,0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000], 'degree':[2,3,4,5,7,9,11,13,15]}

rbf_svm_clf = SVC(class_weight='balanced', kernel='rbf')
rbf_svm_params = {'C': [0.00001,0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000], 'gamma':[0.00001,0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000]}

# classifiers
knn_grid_search = None
linear_grid_search = None
poly_grid_search = None
rbf_grid_search = None


# grid searching
cv = StratifiedKFold(n_splits=4, shuffle=True)

if knn_grid_search == None:
    knn_grid_search = GridSearchCV(knn_clf, knn_params, scoring='average_precision', cv=cv, return_train_score=True)
    knn_grid_search.fit(train_data_normalized, Y)
knn_cv_results = knn_grid_search.cv_results_
# grid search plot
plt.figure()
knn_pivot_table = pd.pivot_table(pd.DataFrame(knn_cv_results),values='mean_test_score', index='param_n_neighbors')
seaborn.heatmap(knn_pivot_table, annot=True).set_title('KNN Grid')

if linear_grid_search == None:
    linear_grid_search = GridSearchCV(linear_svm_clf, linear_svm_params, scoring='average_precision', cv=cv, return_train_score=True)
    linear_grid_search.fit(train_data_normalized, Y)
linear_cv_results = linear_grid_search.cv_results_
# grid search plot
plt.figure()
linear_pivot_table = pd.pivot_table(pd.DataFrame(linear_cv_results),values='mean_test_score', index='param_C')
seaborn.heatmap(linear_pivot_table, annot=True).set_title('Linear SVM Grid')

if poly_grid_search == None:
    poly_grid_search = GridSearchCV(poly_svm_clf, poly_svm_params, scoring='average_precision', cv=cv, return_train_score=True)
    poly_grid_search.fit(train_data_normalized, Y)
poly_cv_results = poly_grid_search.cv_results_
# grid search plot
plt.figure()
poly_pivot_table = pd.pivot_table(pd.DataFrame(poly_cv_results),values='mean_test_score', index='param_C', columns='param_degree')
seaborn.heatmap(poly_pivot_table, annot=True).set_title('Poly SVM Grid')

if rbf_grid_search == None:
    rbf_grid_search = GridSearchCV(rbf_svm_clf, rbf_svm_params, scoring='average_precision', cv=cv, return_train_score=True)
    rbf_grid_search.fit(train_data_normalized, Y)
rbf_cv_results = rbf_grid_search.cv_results_
# grid search plot
plt.figure()
rbf_pivot_table = pd.pivot_table(pd.DataFrame(rbf_cv_results),values='mean_test_score', index='param_C', columns='param_gamma')
seaborn.heatmap(rbf_pivot_table, annot=True).set_title('RBF Grid')

# I selected the rbf based svm as the best classifier based on the heat-map I generated.
# dumping the best classifer to be used for test data
best_classifier = rbf_grid_search.best_estimator_

dump(best_classifier, 'out/clf.dump')
dump(scaler, 'out/min_max_scaler.dump')