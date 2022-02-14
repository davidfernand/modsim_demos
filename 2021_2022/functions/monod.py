"""
Functions for the biochemical reactor with Monod kinetics

Ingmar Nopens, Daan Van Hauwermeiren, David Fernandes del Pozo
"""
# Importer functions and libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# For solver
from scipy.integrate import odeint

# ----------------------------
# Implementation biochemical reactor with Monod kinetics
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{dX}{dt}= \mu_{max}\frac{S}{K_{S}+S}X - \frac{Q}{V}X$$
    $$\frac{dS}{dt}= -\frac{1}{Y} \mu_{max}\frac{S}{K_{S}+S}X + \frac{Q}{V}(S_{in} - S)$$
    '''
    X = variables[0]
    S = variables[1]

    X_new = kwargs['mu_max']*S/(kwargs['K_S'] + S)*X-kwargs['Q']/kwargs['V']*X
    S_new = -kwargs['mu_max']/kwargs['Y']*S/(kwargs['K_S']+S)*X +kwargs['Q']/kwargs['V']*(kwargs['S_in']-S)
    return [X_new, S_new]