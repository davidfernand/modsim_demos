"""
Functions for the force balance model

Ingmar Nopens, Daan Van Hauwermeiren, David Fernandes del Pozo
"""
# Import of all functionalities
import numpy as np
import pandas as pd

from scipy.integrate import odeint

# ----------------------------
# Implementation of the force model with the harmonic oscillator
# ----------------------------

def model_derivatives(variables, t, kwargs):

    x1 = variables[0]
    x2 = variables[1]

    dx1dt = x2
    dx2dt = -kwargs['b']/kwargs['m']*x2 - kwargs['k']/kwargs['m']*x1 + kwargs['Fex']
    return [dx1dt, dx2dt]
