"""
Functions for the respirometry with Double Monod kinetics

David Fernandes del Pozo
"""
# Import functions and libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# For solver
from scipy.integrate import odeint

# ----------------------------
# Implementation of respirometric model with Double Monod kinetics
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{dS1}{dt}= -\frac{\mu_{max,1}X}{Y_1}\frac{S1}{K_{S,1}+S1}-$$
    $$\frac{dS2}{dt}= -\frac{\mu_{max,2}X}{Y_2}\frac{S2}{K_{S,2}+S2}$$
    '''
    S1 = variables[0]
    S2 = variables[1]
    dS1dt = -kwargs['mu_max1']*kwargs['X']/kwargs['Y1']*S1/(kwargs['K_S1']+S1)
    dS2dt = -kwargs['mu_max2']*kwargs['X']/kwargs['Y2']*S2/(kwargs['K_S2']+S2)
    return [dS1dt,dS2dt]