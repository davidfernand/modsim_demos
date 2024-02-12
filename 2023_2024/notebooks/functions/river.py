"""
Functions for the river model

David Fernandes del Pozo
"""
# Import functions and libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# For solver
from scipy.integrate import odeint

# ----------------------------
# Implementation of the river model
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{dBOD}{dt}= BOD_{in}-k_1\cdot BOD$$
    $$\frac{dDO}{dt}=  k_2\cdot (DO_{sat}-DO)-k_1 BOD$$
    '''
    BOD = variables[0]
    DO = variables[1]

    dBODdt = kwargs['BOD_in']-kwargs['k1']*BOD
    dDOdt  = kwargs['k2']*(kwargs['DOsat']-DO)-kwargs['k1']*BOD
    return [dBODdt, dDOdt]
