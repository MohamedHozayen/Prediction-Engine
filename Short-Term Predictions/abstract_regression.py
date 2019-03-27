# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:39:22 2018

@author: MHozayen

ABSTRACT REGRESSION is an interface for testground.py for automation purposes
"""
# Importing the libraries
import numpy as np
import pandas as pd

def model(x, y, model, ws, pred_interval): 

    """
	Model is a function that run predictions model (LR, SVR, RFR, etc
	
	Arguments:
		x: time vector
		y: glucose level vector
		model: type of predictive model - must be imported and has a function predict
            for example: SVR is a model based on support vector regression
                        it has predict function
		ws: Window Size, predict based on the window size
				ws = 6, means predict based on the past half an hour (6 x 5 = 30 minutes)
        pred_interval: predict glucose level after current time (t) in future t+tau
             if pred-intervak is 6 that means prediction for t+6 (next 30 min) baesed on window size
			 

	Returns:
		numpy array of time vector and the corresponding predicted glucose level
	"""
    
    
    # Arrays that will contain predicted values.
    tp_pred = np.zeros(len(y)) 
    yp_pred = np.zeros(len(y))

    for i in range(ws, len(y)-1):
        
        ts_tmp = x[i-ws:i]
        ys_tmp = y[i-ws:i]
          
        if i < x.size-pred_interval:
            #PREDICTION 6 12
            tp = x.iloc[i+pred_interval, 0]
            tp_pred[i] = tp    
            yp_pred[i] = model.predict(ts_tmp, ys_tmp, tp.reshape(-1,1))
            
    tp_pred = np.trim_zeros(tp_pred)
    yp_pred = np.trim_zeros(yp_pred)
    return tp_pred, yp_pred
        

