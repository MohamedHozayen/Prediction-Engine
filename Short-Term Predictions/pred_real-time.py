# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:39:22 2018

@author: MHozayen

PRED_REAL-TIME.py predicts in real time future glucose level. This class represents real life predictions scenarios.
"""
# Importing the libraries
import numpy as np
import pandas as pd
import SVR
import LR
import RFR
import sys, traceback

def pred(x, y, model, ws, pred_interval): 

    """
	Model is a function that run predictions model
    predicts pred_interval based on window size ws
      
    
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
		A single value y-predicted
        
    x = time vector 5 minutes between each point
    y = gloucose level
    
    model = LR, RFR, or SVR
    
    ws =  1, 2, 3, ...
    pred_interval = 1, 2, 3, ...
    
     
    1 is 5 minutes
    2 is 10 minutes
    3 is 15 minutes
    4 is 20 minutes
    5 is 25 minutes
    6 is 30 minutes
    7 is 35 minutes
    
    typical input for ws and pred_interval = 6, 9, 12 for 30, 45, 60 minutes respectively
    ....
    ....
    
	"""
        
    #check if x, and y are the same size
    if len(x) != len(y):
        sys.exit('X and Y must be the same length!')
        
    #check if ws is less than x or y    
    if len(x) < ws:
        sys.exit('Window size ws has to be less than length of X and Y!')
    
    ts_tmp = x[len(x)-ws-1:]
    ys_tmp = y[len(y)-ws-1:]
    tp = x.iloc[len(x)-1, 0] + 5; #5 is hard coded - 5 minutes between points in x
    
    i = 0;
    while i <= pred_interval:
    
        y_predicted = model.predict(ts_tmp, ys_tmp, tp.reshape(-1,1))
        #print(y_predicted)
        tp = tp + 5;
        ts_tmp = ts_tmp.append(pd.DataFrame([tp],))
        #print(ts_tmp)
        if model == LR:
            ys_tmp = ys_tmp.append(pd.DataFrame(y_predicted,))
        else:
            ys_tmp = ys_tmp.append(pd.DataFrame([y_predicted],))
            
        i = i +1;
            
    
    #return the last value in array - accumulative predictions
    return y_predicted[0]       

