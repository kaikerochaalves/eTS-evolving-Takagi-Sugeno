# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:30:20 2023

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
# Importing the Power Trasnformers Datasets
#-----------------------------------------------------------------------------

# Importing the data
Data1 = pd.read_excel (r'Datasets\PowerTransformerDay1.xlsx', header=None)
Data2 = pd.read_excel (r'Datasets\PowerTransformerDay2.xlsx', header=None)
Data3 = pd.read_excel (r'Datasets\PowerTransformerDay3.xlsx', header=None)

# Changing to matrix
Data1 = Data1.to_numpy()
Data2 = Data2.to_numpy()
Data3 = Data3.to_numpy()
# Separating the inputs and output
X1 = Data1[:-1,:-1]
y1 = Data1[:-1,-1]

X2 = Data2[:-1,:-1]
y2 = Data2[:-1,-1]

X3 = Data3[:-1,:-1]
y3 = Data3[:-1,-1]

def Normalize_Data(data):
    Max_data = np.max(data)
    Min_data = np.min(data)
    Normalized_data = (data - Min_data)/(Max_data - Min_data)
    return Normalized_data, Max_data, Min_data

# Use the first day to train and the second and third to test
X_train = X1
y_train = y1

# Normalize the inputs and the output
Normalized_X_train, X_train_max, X_train_min = Normalize_Data(X_train)
Normalized_y_train, y_train_max, y_train_min = Normalize_Data(y_train)

# Use the first day to train and the second and third to test
X_test1 = X2
y_test1 = y2

# Normalize the inputs and the output for day 2
Normalized_X_test1, X_test1_max, X_test1_min = Normalize_Data(X_test1)
Normalized_y_test1, y_test1_max, y_test1_min = Normalize_Data(y_test1)

X_test2 = X3
y_test2 = y3

# Normalize the inputs and the output for day 3
Normalized_X_test2, X_test2_max, X_test2_min = Normalize_Data(X_test2)
Normalized_y_test2, y_test2_max, y_test2_min = Normalize_Data(y_test2)

# Plotting rules' evolution of the model
plt.plot(y_train, color='blue')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.show()

# Plotting rules' evolution of the model
plt.plot(y_test1, color='blue')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.show()

# Plotting rules' evolution of the model
plt.plot(y_test2, color='blue')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.show()

#-----------------------------------------------------------------------------
# Calling the eTS
#-----------------------------------------------------------------------------

# Day 2

# Setting the hyperparameters
InitialOmega = 1000
r = 0.1

# Initialize the model
model = eTS(InitialOmega = InitialOmega, r = r)
# Train the model
OutputTraining1, Rules1 = model.fit(Normalized_X_train, Normalized_y_train)
# Test the model with day 2
OutputTest1 = model.predict(Normalized_X_test1)

# Day 3

# Setting the hyperparameters
InitialOmega = 1000
r = 0.2

# Initialize the model
model = eTS(InitialOmega = InitialOmega, r = r)
# Train the model
OutputTraining2, Rules2 = model.fit(Normalized_X_train, Normalized_y_train)
# Test the model with day 3
OutputTest2 = model.predict(Normalized_X_test2)

#-----------------------------------------------------------------------------
# Evaluate the model's performance
#-----------------------------------------------------------------------------

# Day 2

# Calculating the error metrics
# DeNormalize the results
OutputTestDenormalized1 = OutputTest1 * (y_test1_max - y_test1_min) + y_test1_min
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test1, OutputTestDenormalized1))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test1.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test1, OutputTestDenormalized1)

# Printing the RMSE
print("RMSE = ", RMSE)
# Printing the NDEI
print("NDEI = ", NDEI)
# Printing the MAE
print("MAE = ", MAE)
# Printing the number of final rules
print("Final Rules = ", Rules1[-1])

# Day 3

# Calculating the error metrics
# DeNormalize the results
OutputTestDenormalized2 = OutputTest2 * (y_test2_max - y_test2_min) + y_test2_min
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test2, OutputTestDenormalized2))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test2.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test2, OutputTestDenormalized2)

# Printing the RMSE
print("\nRMSE = ", RMSE)
# Printing the NDEI
print("NDEI = ", NDEI)
# Printing the MAE
print("MAE = ", MAE)
# Printing the number of final rules
print("Final Rules = ", Rules2[-1])

#-----------------------------------------------------------------------------
# Plot the graphics
#-----------------------------------------------------------------------------

# Day 2

# Plot the graphic of the actual time series and the eTS predictions
plt.plot(y_test1, label='Actual Value', color='red')
plt.plot(OutputTestDenormalized1, color='blue', label='eTS')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend()
plt.show()

# Plot the evolution of the model's rule
plt.plot(Rules1, color='blue')
plt.ylabel('Number of Fuzzy Rules')
plt.xlabel('Samples')
plt.show()

# Day 3

# Plot the graphic of the actual time series and the eTS predictions
plt.plot(y_test2, label='Actual Value', color='red')
plt.plot(OutputTestDenormalized2, color='blue', label='eTS')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend()
plt.show()

# Plot the evolution of the model's rule
plt.plot(Rules2, color='blue')
plt.ylabel('Number of Fuzzy Rules')
plt.xlabel('Samples')
plt.show()