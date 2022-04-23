# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:32:35 2022

@author: kaike
"""

import numpy as np
import math

def Nonlinear(sample_n = 10000):
    
    X = np.zeros(sample_n)
    u = np.zeros(sample_n - 1)
    
    u[0] = math.sin(2*math.pi*1/25)
    X[0] = 0
    X[1] = 0
  
    for i in range(2, sample_n):
        u[i-1] = math.sin(2 * math.pi * i / 25)
        X[i] = ((X[i-1] * X[i-2] * (X[i-1] - 0.5)) / (1 + X[i-1]**2 + X[i-2]**2)) + u[i-1]
        
    return X, u