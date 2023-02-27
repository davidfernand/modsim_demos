"""
Functions for the nitrobenzene example

Ingmar Nopens, Daan Van Hauwermeiren, David Fernandes del Pozo
"""
# Import all relevant functions and libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# For the solver
from scipy.integrate import odeint

# ----------------------------
# Implementation of the nitrobenzene model
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{d[C_6H_6]}{dt}= -2k{[C_6H_6]}^2[N_2O_5] + \frac{Q}{V}({[C_6H_6]}_{in} - [C_6H_6])$$
    $$\frac{d[N_2O_5]}{dt}= -k{[C_6H_6]}^2[N_2O_5] + \frac{Q}{V}({[N_2O_5]}_{in} - [N_2O_5])$$
    $$\frac{d[C_6H_5NO_2]}{dt}= 2k{[C_6H_6]}^2[N_2O_5] - \frac{Q}{V}[C_6H_5NO_2] $$
    '''
    C6H6 = variables[0]
    N2O5 = variables[1]
    C6H5NO2 = variables[2]

    dC6H6dt = -2*kwargs['k']*C6H6*C6H6*N2O5 + kwargs['Q']/kwargs['V']*(kwargs['C6H6_in'] - C6H6)
    dN2O5dt = -kwargs['k']*C6H6*C6H6*N2O5 + kwargs['Q']/kwargs['V']*(kwargs['N2O5_in'] - N2O5)
    dC6H5NO2dt = 2*kwargs['k']*C6H6*C6H6*N2O5 - kwargs['Q']/kwargs['V']*C6H5NO2
    return [dC6H6dt, dN2O5dt, dC6H5NO2dt]
