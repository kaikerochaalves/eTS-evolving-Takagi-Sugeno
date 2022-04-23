# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:46:28 2022

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

# Importing the library to generate the Mackey Glass time series
from MackeyGlassGenerator import MackeyGlass

# Importing the library to generate the Mackey Glass time series
from NonlinearGenerator import Nonlinear

# Importing the library to generate the Mackey Glass time series
from LorenzAttractorGenerator import Lorenz

# Defining the atributes
from statsmodels.tsa.tsatools import lagmat

# Importing the model
from eTS import eTS


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

#-----------------------------------------------------------------------------
# Define the search space for the hyperparameters
#-----------------------------------------------------------------------------

l_InitialOmega = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
l_r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 5, 10, 20, 30, 40, 50]


#-----------------------------------------------------------------------------
# Generating the Mackey-Glass time series
#-----------------------------------------------------------------------------

# The theory
# Mackey-Glass time series refers to the following, delayed differential equation:
    
# dx(t)/dt = ax(t-\tau)/(1 + x(t-\tau)^10) - bx(t)


# Input parameters
a        = 0.2;     # value for a in eq (1)
b        = 0.1;     # value for b in eq (1)
tau      = 17;		# delay constant in eq (1)
x0       = 1.2;		# initial condition: x(t=0)=x0
sample_n = 6000;	# total no. of samples, excluding the given initial condition

# MG = mackey_glass(N, a = a, b = b, c = c, d = d, e = e, initial = initial)
MG = MackeyGlass(a = a, b = b, tau = tau, x0 = x0, sample_n = sample_n)

# Defining the atributes and the target value
X, y = Create_Leg(MG, ncols = 4, leg = 6, leg_output = 85)

# Spliting the data into train and test
X_train, X_test = X[201:3201,:], X[5001:5501,:]
y_train, y_test = y[201:3201,:], y[5001:5501,:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[201:3201,:], Normalized_X[5001:5501,:]
Normalized_y_train, Normalized_y_test = Normalized_y[201:3201,:], Normalized_y[5001:5501,:]

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Mackey-Glass time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        # Test the model
        OutputTest = model.predict(Normalized_X_test)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTestDenormalized = OutputTest * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_test, OutputTestDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_test.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_test, OutputTestDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_MackeyGlass.xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Generating the Nonlinear time series
#-----------------------------------------------------------------------------
    
sample_n = 6000
NTS, u = Nonlinear(sample_n)       

# Defining the atributes and the target value
X, y = Create_Leg(NTS, ncols = 2, leg = 1, leg_output = 1)
X = np.append(X, u[:X.shape[0]].reshape(-1,1), axis = 1)

# Spliting the data into train and test
X_train, X_test = X[2:5002,:], X[5002:5202,:]
y_train, y_test = y[2:5002,:], y[5002:5202,:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[2:5002,:], Normalized_X[5002:5202,:]
Normalized_y_train, Normalized_y_test = Normalized_y[2:5002,:], Normalized_y[5002:5202,:]

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Nonlinear time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        # Test the model
        OutputTest = model.predict(Normalized_X_test)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTestDenormalized = OutputTest * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_test, OutputTestDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_test.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_test, OutputTestDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_Nonlinear.xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Generating the Lorenz Attractor time series
#-----------------------------------------------------------------------------


# Input parameters
x0 = 0.
y0 = 1.
z0 = 1.05
sigma = 10
beta = 2.667
rho=28
num_steps = 10000

# Creating the Lorenz Time Series
x, y, z = Lorenz(x0 = x0, y0 = y0, z0 = z0, sigma = sigma, beta = beta, rho = rho, num_steps = num_steps)

# Defining the atributes and the target value
X = np.concatenate([x[:-1].reshape(-1,1), y[:-1].reshape(-1,1), z[:-1].reshape(-1,1)], axis = 1)
y = x[1:].reshape(-1,1)

# Spliting the data into train and test
X_train, X_test = X[:8000,:], X[8000:,:]
y_train, y_test = y[:8000,:], y[8000:,:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[:8000,:], Normalized_X[8000:,:]
Normalized_y_train, Normalized_y_test = Normalized_y[:8000,:], Normalized_y[8000:,:]

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Lorenz Attractor time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        # Test the model
        OutputTest = model.predict(Normalized_X_test)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTestDenormalized = OutputTest * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_test, OutputTestDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_test.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_test, OutputTestDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_LorenzAttractor.xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Importing the NASDAQ time series
#-----------------------------------------------------------------------------
    
# Importing the data
NASDAQ = pd.read_csv(r'Datasets\NASDAQ.csv')

Close = NASDAQ['Adj Close'].values
samples_n = Close.shape[0]
training_size = round(0.8 * samples_n)

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

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the NASDAQ time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        # Test the model
        OutputTest = model.predict(Normalized_X_test)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTestDenormalized = OutputTest * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_test, OutputTestDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_test.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_test, OutputTestDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_NASDAQ.xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Importing the SP500 time series
#-----------------------------------------------------------------------------
    
# Importing the data
SP500 = pd.read_csv(r'Datasets\SP500.csv')

Close = SP500['Close'].values
samples_n = Close.shape[0]
training_size = round(0.8 * samples_n)


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

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the SP500 time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        # Test the model
        OutputTest = model.predict(Normalized_X_test)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTestDenormalized = OutputTest * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_test, OutputTestDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_test.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_test, OutputTestDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_SP500.xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Importing the TAIEX time series
#-----------------------------------------------------------------------------
    
# Importing the data
TAIEX = pd.read_csv(r'Datasets\TAIEX.csv')

Close = TAIEX['Close'].values
samples_n = Close.shape[0]
training_size = round(0.8 * samples_n)

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

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the TAIEX time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        # Test the model
        OutputTest = model.predict(Normalized_X_test)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTestDenormalized = OutputTest * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_test, OutputTestDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_test.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_test, OutputTestDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_TAIEX .xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Importing the Power Trasnformers Dataset Day 1
#-----------------------------------------------------------------------------

# Importing the data
Data = pd.read_excel (r'Datasets\PowerTransformerDay1.xlsx', header=None)

# Total number of points
n = Data.shape[0]
# Defining the number of training points
training_size = n


# Changing to matrix
Data = Data.to_numpy()
# Separating the inputs and output
X = Data[:-1,:-1]
y = Data[:-1,-1]

# Spliting the data into train and test
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[:training_size,:], Normalized_X[training_size:,:]
Normalized_y_train, Normalized_y_test = Normalized_y[:training_size], Normalized_y[training_size:]

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Power Trasnformers Dataset Day 1
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTrainingDenormalized = OutputTraining * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_train, OutputTrainingDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_train.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_train, OutputTrainingDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_TransformerDay1.xlsx"
result.to_excel(name)

#-----------------------------------------------------------------------------
# Importing the Power Trasnformers Dataset Day 2
#-----------------------------------------------------------------------------

# Importing the data
Data = pd.read_excel (r'Datasets\PowerTransformerDay2.xlsx', header=None)

# Total number of points
n = Data.shape[0]
# Defining the number of training points
training_size = n


# Changing to matrix
Data = Data.to_numpy()
# Separating the inputs and output
X = Data[:-1,:-1]
y = Data[:-1,-1]

# Spliting the data into train and test
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[:training_size,:], Normalized_X[training_size:,:]
Normalized_y_train, Normalized_y_test = Normalized_y[:training_size], Normalized_y[training_size:]

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Power Trasnformers Dataset Day 2
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTrainingDenormalized = OutputTraining * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_train, OutputTrainingDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_train.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_train, OutputTrainingDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_TransformerDay2.xlsx"
result.to_excel(name)

#-----------------------------------------------------------------------------
# Importing the Power Trasnformers Dataset Day 3
#-----------------------------------------------------------------------------

# Importing the data
Data = pd.read_excel (r'Datasets\PowerTransformerDay3.xlsx', header=None)

# Total number of points
n = Data.shape[0]
# Defining the number of training points
training_size = n


# Changing to matrix
Data = Data.to_numpy()
# Separating the inputs and output
X = Data[:-1,:-1]
y = Data[:-1,-1]

# Spliting the data into train and test
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Normalize the inputs and the output
Normalized_X, X_max, X_min = Normalize_Data(X)
Normalized_y, y_max, y_min = Normalize_Data(y)

# Spliting normilized data into train and test
Normalized_X_train, Normalized_X_test = Normalized_X[:training_size,:], Normalized_X[training_size:,:]
Normalized_y_train, Normalized_y_test = Normalized_y[:training_size], Normalized_y[training_size:]

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Power Trasnformers Dataset Day 3
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
result = pd.DataFrame(columns = ['InitialOmega', 'r', 'RMSE', 'NDEI', 'MAE', 'Rules'])
for InitialOmega in l_InitialOmega:
    for r in l_r:
        
        # Initialize the model
        model = eTS(InitialOmega = InitialOmega, r = r)
        # Train the model
        OutputTraining, Rules = model.fit(Normalized_X_train, Normalized_y_train)
        
        # Calculating the error metrics
        # DeNormalize the results
        OutputTrainingDenormalized = OutputTraining * (y_max - y_min) + y_min
        # Compute the Root Mean Square Error
        RMSE = math.sqrt(mean_squared_error(y_train, OutputTrainingDenormalized))
        # Compute the Non-Dimensional Error Index
        NDEI= RMSE/st.stdev(y_train.flatten())
        # Compute the Mean Absolute Error
        MAE = mean_absolute_error(y_train, OutputTrainingDenormalized)
        # Compute the number of final rules
        Rules = Rules[-1]
        
        simul = simul + 1
        print(f'Simulação: {simul}')
        
        result = result.append({'InitialOmega': InitialOmega, 'r': r, 'RMSE': RMSE, 'NDEI': NDEI, 'MAE': MAE, 'Rules': Rules}, ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_TransformerDay3.xlsx"
result.to_excel(name)