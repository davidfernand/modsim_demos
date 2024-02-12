"""
Functions for the river model by specifying 
a Monod model for the biological degradation

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
    $$\frac{dBOD}{dt}= BOD_{in}-k_{max,1}\cdot \frac{BOD}{BOD+K_{S,1}}$$
    $$\frac{dDO}{dt}=  k_2\cdot (DO_{sat}-DO)-k_{max,1}\cdot \frac{BOD}{BOD+K_{S,1}}$$
    '''
    BOD = variables[0]
    DO = variables[1]

    dBODdt = kwargs['BOD_in']-kwargs['kmax1']*BOD/(kwargs['K_S1']+BOD)*BOD
    dDOdt  = kwargs['k2']*(kwargs['DOsat']-DO)-kwargs['kmax1']*BOD/(kwargs['K_S1']+BOD)*BOD
    return [dBODdt, dDOdt]
