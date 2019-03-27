# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:00:59 2019

@author: MHozayen

Random Forest Regression
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
	Predict is an interface for abstract_regression.py
		predicts one point (pred) based on x and y vecotr using RFR

		Arguments:
		x: time vector
		y: glucose level vector 
		pred: is the time in future corrosponding to the glucose level of interest
		
	Returns:
		Flaot value of the predicted glucose level at t time (pred)
"""
def predict(x, y, pred):
     #degree is unused here

   
    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 1, random_state = 0, min_samples_split = 2)
    regressor.fit(x, y)
    
    # Predicting a new result
    y_pred = regressor.predict(pred)
    
    return y_pred