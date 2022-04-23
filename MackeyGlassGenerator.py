# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:32:35 2022

@author: kaike
"""

import numpy as np
import math

def MackeyGlass_DifferentialEquation(x_t, x_t_minus_tau, a, b):
    """
    
    This function returns dx/dt of Mackey-Glass delayed differential equation:
        
    dx(t)/dt = ax(t-\tau)/(1 + x(t-\tau)^10) - bx(t)

    """
    
    x_dot = - b * x_t + a * x_t_minus_tau/(1 + x_t_minus_tau**10.0)
    
    return x_dot

def MackeyGlass_RK4(x_t, x_t_minus_tau, deltat, a, b):
    """
    This function computes the numerical solution of the Mackey-Glass
    delayed differential equation using the 4-th order Runge-Kutta method

    """
    
    k1 = deltat * MackeyGlass_DifferentialEquation(x_t,            x_t_minus_tau, a, b)
    k2 = deltat * MackeyGlass_DifferentialEquation(x_t + 0.5 * k1, x_t_minus_tau, a, b)
    k3 = deltat * MackeyGlass_DifferentialEquation(x_t + 0.5 * k2, x_t_minus_tau, a, b)
    k4 = deltat * MackeyGlass_DifferentialEquation(x_t + k3,       x_t_minus_tau, a, b)
    x_t_plus_deltat = (x_t + k1/6 + k2/3 + k3/3 + k4/6)
    return x_t_plus_deltat

def MackeyGlass(a = 0.2, b = 0.1, tau = 17, x0 = 1.2, sample_n = 10000):
    time = 0
    index = 1
    deltat = 1 #  time step size (which coincides with the integration step)
    history_length = math.floor(tau/deltat)
    x_history = np.zeros(history_length) # here we assume x(t)=0 for -tau <= t < 0
    x_t = x0
    
    X = np.zeros(sample_n + 1) # vector of all generated x samples
    T = np.zeros(sample_n + 1) # vector of time samples
    
    for i in range(sample_n + 1):
        X[i] = x_t
        if tau == 0:
            x_t_minus_tau = 0.0
        else:
            x_t_minus_tau = x_history[index-1]
            
        x_t_plus_deltat = MackeyGlass_RK4(x_t, x_t_minus_tau, deltat, a, b)
        
        if tau != 0:
            x_history[index-1] = x_t_plus_deltat
            index = (index % history_length) + 1
            
        time = time + deltat
        T[i] = time
        x_t = x_t_plus_deltat
        
    return X