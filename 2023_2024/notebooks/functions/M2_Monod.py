"""
Functions for the respirometry with Monod kinetics

David Fernandes del Pozo
"""
# Import functions and libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# For solver
from scipy.integrate import odeint

# ----------------------------
# Implementation of respirometric model with Monod kinetics
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{dS}{dt}= -\frac{\mu_{max,1}X}{Y_1}\frac{S}{K_{S,1}+S}$$
    '''
    S = variables
    dSdt = -kwargs['mu_max']*kwargs['X']/kwargs['Y']*S/(kwargs['K_S']+S)
    return dSdt 