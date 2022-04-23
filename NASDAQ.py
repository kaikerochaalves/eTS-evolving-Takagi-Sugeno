# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:52:50 2022

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""

# Importing libraries
import math
import numpy as np
import pandas as pd
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt 

# Defining the atributes
from statsmodels.tsa.tsatools import lagmat

# Importing the model
from eTS import eTS

#-----------------------------------------------------------------------------
# Importing the NASDAQ time series
#-----------------------------------------------------------------------------
    
# Importing the data
NASDAQ = pd.read_csv(r'Datasets\NASDAQ.csv')

Close = NASDAQ['Adj Close'].values
samples_n = Close.shape[0]
training_size = round(0.8 * samples_n)

def Create_Leg(data, ncols, leg, leg_output = None):
    X = np.array(data[leg*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[leg*i:leg*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if leg_output == None:
        return X_new
    else:
        y = np.array(data[leg*(ncols-1)+leg_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y
    
def Normalize_Data(data):
    Max_data = np.max(data)
    Min_data = np.min(data)
    Normalized_data = (data - Min_data)/(Max_data - Min_data)
    return Normalized_data, Max_data, Min_data
        

# Defining the atributes and the target value
X, y = Create_Leg(Close, ncols = 3, leg = 1, leg_output = 1)


# Spliting the data into train and test
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size,:], y[training_size:,:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[:training_size,:], Normalized_X[training_size:,:]
Normalized_y_train, Normalized_y_test = Normalized_y[:training_size,:], Normalized_y[training_size:,:]


# Plotting rules' evolution of the model
plt.plot(y_test, color='blue')
plt.ylabel('Close')
plt.xlabel('Samples')
plt.show()

#-----------------------------------------------------------------------------
# Calling the eTS
#-----------------------------------------------------------------------------

# Setting the hyperparameters
InitialOmega = 1000
r = 0.5

# Initialize the model
model = eTS(InitialOmega = InitialOmega, r = r)
# Train the model
OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
# Test the model
OutputTest = model.predict(Normalized_X_test)

#-----------------------------------------------------------------------------
# Evaluate the model's performance
#-----------------------------------------------------------------------------

# Calculating the error metrics
# DeNormalize the results
OutputTestDenormalized = OutputTest * (y_max - y_min) + y_min
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, OutputTestDenormalized))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, OutputTestDenormalized)

# Printing the RMSE
print("RMSE = ", RMSE)
# Printing the NDEI
print("NDEI = ", NDEI)
# Printing the MAE
print("MAE = ", MAE)
# Printing the number of final rules
print("Final Rules = ", Rules[-1])

#-----------------------------------------------------------------------------
# Plot the graphics
#-----------------------------------------------------------------------------

# Plot the graphic of the actual time series and the eTS predictions
plt.plot(y_test, label='Actual Value', color='red')
plt.plot(OutputTestDenormalized, color='blue', label='eTS')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend()
plt.show()

# Plot the evolution of the model's rule
plt.plot(Rules, color='blue')
plt.ylabel('Number of Fuzzy Rules')
plt.xlabel('Samples')
plt.show()