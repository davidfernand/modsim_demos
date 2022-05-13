"""
Functions for the river-contamination case

David Fernandes del Pozo
"""
# Import functions and libraries
import numpy as np

# ----------------------------
# Implementation of river case model
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{d BOD}{d t} = BOD_{in} - k_1 \cdot BOD$$
    $$\frac{d DO}{d t} = k_2 \cdot (DO_{sat} - DO) - k_1 \cdot BOD$$
    '''
    BOD = variables[0]
    DO = variables[1]

    dBODdt = kwargs['BODin']-kwargs['k1']*BOD                   # BOD (mg/L)
    dDOdt =  kwargs['k2']*(kwargs['DOsat']-DO)-kwargs['k1']*BOD # DO (mg/L)
    return [ dBODdt, dDOdt]
