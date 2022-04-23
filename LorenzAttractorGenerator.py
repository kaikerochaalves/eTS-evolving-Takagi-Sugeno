# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 01:38:11 2022

@author: kaike
"""

import numpy as np
import matplotlib.pyplot as plt


def LorenzAttractor_DifferentialEquation(x, y, z, sigma = 10, beta = 2.667, rho=28):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = sigma * (y - x)
    y_dot = rho*x - y - x*z
    z_dot = x*y - beta*z
    return x_dot, y_dot, z_dot

def Lorenz(x0 = 0., y0 = 1., z0 = 1.05, sigma = 10, beta = 2.667, rho=28, num_steps = 10000):

    dt = 0.01
    
    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    
    # Set initial values
    xs[0], ys[0], zs[0] = (x0, y0, z0)
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = LorenzAttractor_DifferentialEquation(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        
    return xs, ys, zs