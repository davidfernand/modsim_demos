"""
Functions for the reactor model with temperature dependent kinetics

Ingmar Nopens, Daan Van Hauwermeiren, David Fernandes del Pozo
"""
# Importing functionalities
import numpy as np
import pandas as pd
import math

# For solver
from scipy.integrate import odeint

# ----------------------------
# Implementation of heat model for temperature as a variable
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{d[N_2O_5]}{dt}= -2A_re^{\frac{-E_{A}}{RT}}[N_2O_5] + \frac{Q}{V}({[N_2O_5]}_{in} - [N_2O_5])$$
    $$\frac{d[N_2O_4]}{dt}= 2A_re^{\frac{-E_{A}}{RT}}[N_2O_5]+ \frac{Q}{V}({[N_2O_4]}_{in} - [N_2O_4])$$
    $$\frac{d[T]}{dt}= \frac{Q \rho C_{p}(T_{in}-T)+UA(T_{w}-T)-VA_re^{\frac{-E_{A}}{RT}}[N_2O_5]\Delta_{r}H}{V \rho C_{p}}$$
    '''
    N2O5 = variables[0]
    N2O4 = variables[1]
    T = variables[2]

    R = 8.314
    dN2O5dt = -2*kwargs['Ar']*math.exp(-kwargs['Ea']/(R*T))*N2O5 \
              + kwargs['Q']/kwargs['V']*(kwargs['N2O5_in'] - N2O5)
    dN2O4dt = 2*kwargs['Ar']*math.exp(-kwargs['Ea']/(R*T))*N2O5 \
              +kwargs['Q']/kwargs['V']*(kwargs['N2O4_in'] - N2O4)
    dTdt = kwargs['Q']/kwargs['V']*(kwargs['Tin'] - T) \
          +kwargs['U']*kwargs['A']/(kwargs['V']*kwargs['rho']*kwargs['Cp']) \
          *(kwargs['Tw'] - T) \
          -kwargs['Ar']/(kwargs['V']*kwargs['rho']*kwargs['Cp'])\
          *math.exp(-kwargs['Ea']/(R*T))*N2O5*kwargs['delta_rH']
    return [dN2O5dt, dN2O4dt, dTdt]
