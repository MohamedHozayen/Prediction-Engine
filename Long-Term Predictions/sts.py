# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:25:16 2018

@author: Jason Brownlee
@author: Mohamed Hozayen

The code below uses the function series_to_supervised which was devloped by Jason BrownLee (PhD)
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
"""
#from pandas import DataFrame
#from pandas import concat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame
from pandas import concat
 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        #names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        names += [('y(t-%d)' % (i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += ['y(t)'  for j in range(n_vars)]
            #names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Importing the dataset
dataset = pd.read_csv('Data//data-points_cases.csv')
# Healthy glucose subjects
gl_h = dataset.iloc[19:, 1:7].values
#gl_h = pd.DataFrame(gl_h)
gl_h = gl_h.astype('U').astype(np.float) 
# Diabatic glucos subjects
gl_d = dataset.iloc[19:, 7:].values
#gl_d = pd.DataFrame(gl_d)
gl_d = gl_d.astype('U').astype(np.float)


ys = pd.DataFrame(gl_h[:,0])
data = series_to_supervised(ys, 3)
data1 = series_to_supervised(ys, 1)
data2 = series_to_supervised(ys, 2)
data3 = series_to_supervised(ys, 3)
data4 = series_to_supervised(ys, 4)
data7 = series_to_supervised(ys, 7)


#time axis
dif = 5 # 5 min among data points
n_dif = 60/5 #how many 5 min in 1 hour
t = np.linspace(0, dif*12*24, np.size(gl_h, 0))
plt.plot(t, ys, color = 'red') # , label='Diabetic'
plt.title('Glucose Level Over Time')
plt.xlabel('Time')
plt.ylabel('Glucose Level (mmol/L)')
plt.show()

