import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
#
from sklearn.linear_model import LinearRegression

x_data = np.load('regression_x_data.npy')
y_data = np.load('regression_y_data.npy')
print(x_data.shape) #(40, 1)
print(y_data.shape) #(40,)
print(x_data[:2])
'''
[[2.45265086]
[2.09839213]]
'''
print(y_data[:2]) #[-0.82075445  0.86401998]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data)

x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train)

#estimator = LinearRegression()
estimator = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

estimator.fit(x_train, y_train)

y_predict = estimator.predict(x_val)
score = metrics.r2_score(y_val, y_predict) #regression
#score = metrics.accuracy_score(y_val, y_predict) #classification
#score = metrics.roc_auc_score(y_val, y_predict) #classification
print(score) #1.0
