"""
Functions for the respirometry with exponential kinetics

David Fernandes del Pozo
"""
# Import functions and libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# For solver
from scipy.integrate import odeint

# ----------------------------
# Implementation of respirometric model with exponential kinetics
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{dS}{dt}= -\frac{\mu_{max}X}{Y_1}S$$
    '''
    S = variables
    dSdt = -kwargs['mu_max']*kwargs['X']/kwargs['Y']*S
    return dSdt 