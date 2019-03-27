# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:00:59 2019

@author: MHozayen

Support Vector Regresssion
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
	Predict is an interface for abstract_regression.py
		predicts one point (pred) based on x and y vecotr using SVR
		
	Arguments:
		x: time vector
		y: glucose level vector 
		pred: is the time in future corrosponding to the glucose level of interest
		
	Returns:
		Flaot value of the predicted glucose level at t time (pred)
"""

def predict(x, y, pred):

    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(x)
    Y = sc_y.fit_transform(y)
    
    # Fitting SVR to the dataset
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, Y)
    
    # Predicting a new result
    y_pred = regressor.predict(pred)
    y_pred = sc_y.inverse_transform(y_pred)
    return y_pred