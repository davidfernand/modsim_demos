"""
Functions used for the population model

Ingmar Nopens, Daan Van Hauwermeiren, David Fernandes del Pozo
"""
# Import all relevant python functions and libraries
import matplotlib.pyplot as plt
#import seaborn as  sns
import numpy as np
import pandas as pd
# For our solver
from scipy.integrate import odeint

# ----------------------------
# Implementation of the population model
# ----------------------------

def model_derivatives(variables, t, kwargs):
    '''
    $$\frac{dv}{dt}= r_{v}v (1-\frac{v}{K_{v}}) - d_{nv}v$$
    $$\frac{dm_{1}}{dt}=r_{1}m_{1}(1-\frac{m_{1}+m_{2}}{K_{M}})-\alpha_{1}vm_{1}-d_{n1}m_{1}$$
    $$\frac{dm_{2}}{dt}=r_{2}m_{2}(1-\frac{m_{1}+m_{2}}{K_{M}})-\alpha_{2}vm_{2}-d_{n2}m_{2}+m_{2,in}$$
    '''
    v = variables[0]
    m1 = variables[1]
    m2 = variables[2]

    dvdt = kwargs['r_v']*v*(1-v/kwargs['K_v']) - kwargs['d_nv']*v
    dm1dt = kwargs['r_1']*m1*(1-(m1+m2)/kwargs['K_m'])-kwargs['alpha_1']*v*m1-kwargs['d_n1']*m1
    dm2dt = kwargs['r_2']*m2*(1-(m1+m2)/kwargs['K_m'])-kwargs['alpha_2']*v*m2-kwargs['d_n2']*m2+kwargs['m2_in']
    return [dvdt, dm1dt, dm2dt]
