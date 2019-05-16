# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:37:20 2019
@author: Muhammad Shahbaz (MS-18-IT-507815)
"""
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
import seaborn
from joblib import load, dump
from matplotlib import pyplot as plt
import pandas as pd

# a lot of warnings
import warnings
warnings.filterwarnings('ignore')

"""
option 1 => uses the full length audio training data
option 2 => uses the individual clip (4 sec chunks of every audio) training data
"""
option = 1
# loading data
data = None

if option == 1:
    data = load('out/full_length_data_pca_reduced.dump')
else:
    data = load('out/clip_based_data_pca_reduced.dump')
Xtr = data[:,1:-1]
Ytr = data[:,-1]

# checking if data is normalized or not by plotting ranges
number_of_points, dimensions = Xtr.shape

x_ranges = []
for i in range(dimensions): x_ranges.append([Xtr[:,i].min(), Xtr[:,i].max()])
for i, xr in enumerate(x_ranges): plt.plot(xr, [i+1, i+1], 'ro-')
plt.title('before normalization')
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(Xtr)
normalXtr = scaler.transform(Xtr)

x_ranges = []
for i in range(dimensions): x_ranges.append([normalXtr[:,i].min(), normalXtr[:,i].max()])
for i, xr in enumerate(x_ranges): plt.plot(xr, [i+1, i+1], 'ro-')
plt.title('after normalization')
plt.show()

# checking class distribution
counts = dict(Counter(Ytr))
print ('label counts:', counts)
print ('data is not too much imbalanced')

# defining classifiers to be used in grid search
knn_clf = KNeighborsClassifier()
knn_params = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13, 15]}

linear_svm_clf = LinearSVC(class_weight='balanced')
linear_svm_params = {'C':[1e-10, 1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

poly_svm_clf = SVC(class_weight='balanced', kernel='poly')
poly_svm_params = {'C': [1e-10, 1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'degree':[2,3,4,5,7,9,11,13,15]}

rbf_svm_clf = SVC(class_weight='balanced', kernel='rbf')
rbf_svm_params = {'C': [1e-10, 1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'gamma':[0.00001,0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000]}

# cross validations
from leave_one_person_out_cv import lopo_cross_validation
cv = lopo_cross_validation(data)
# scoring
scoring = "accuracy"

# knn
knn_grid_search = GridSearchCV(knn_clf, knn_params, scoring=scoring, cv=cv, return_train_score=True)
knn_grid_search.fit(normalXtr, Ytr)

# grid search plot
plt.figure()
knn_pivot_table = pd.pivot_table(pd.DataFrame(knn_grid_search.cv_results_),values='mean_test_score', index='param_n_neighbors')
seaborn.heatmap(knn_pivot_table, annot=True).set_title('KNN Grid')

# linear
linear_grid_search = GridSearchCV(linear_svm_clf, linear_svm_params, scoring=scoring, cv=cv, return_train_score=True)
linear_grid_search.fit(normalXtr, Ytr)

# grid search plot
plt.figure()
linear_pivot_table = pd.pivot_table(pd.DataFrame(linear_grid_search.cv_results_),values='mean_test_score', index='param_C')
seaborn.heatmap(linear_pivot_table, annot=True).set_title('Linear SVM Grid')

# poly
poly_grid_search = GridSearchCV(poly_svm_clf, poly_svm_params, scoring=scoring, cv=cv, return_train_score=True)
poly_grid_search.fit(normalXtr, Ytr)

# grid search plot
plt.figure()
poly_pivot_table = pd.pivot_table(pd.DataFrame(poly_grid_search.cv_results_),values='mean_test_score', index='param_C', columns='param_degree')
seaborn.heatmap(poly_pivot_table, annot=True).set_title('Poly SVM Grid')

rbf_grid_search = GridSearchCV(rbf_svm_clf, rbf_svm_params, scoring=scoring, cv=cv, return_train_score=True)
rbf_grid_search.fit(normalXtr, Ytr)

# grid search plot
plt.figure()
rbf_pivot_table = pd.pivot_table(pd.DataFrame(rbf_grid_search.cv_results_),values='mean_test_score', index='param_C', columns='param_gamma')
seaborn.heatmap(rbf_pivot_table, annot=True).set_title('RBF Grid')

# dumping the scaler and best classifer to be used for test data
if option == 1:
    dump(scaler, 'out/full_length_data_min_max_scaler.dump')
    dump(linear_grid_search.best_estimator_, 'out/full_length_data_clf.dump')
else:
    dump(scaler, 'out/clip_based_data_min_max_scaler.dump')
    dump(rbf_grid_search.best_estimator_, 'out/clip_based_data_clf.dump')