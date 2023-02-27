"""
Functions for different steps to describe first order dynamics and a second order tank dynamics

David Fernandes del Pozo
"""

import numpy as np

# ----------------------------
# Implementation of different steps to describe first order dynamics and a second order tank dynamics
# ----------------------------


def impulse(t,tau,k,y_0):
    return np.exp(-t/tau)*y_0 + k/tau*np.exp(-t/tau)

def step(t,tau,k,y_0):
    return np.exp(-t/tau)*y_0 + k*(1-np.exp(-t/tau)) 
    
def ramp(t,tau,k,y_0):
    return np.exp(-t/tau)*y_0 + k*tau*np.exp(-t/tau)+k*(t-tau)

def freq_sine_wave(t,tau,k,a,omega,y_0):
    c1=k*a*omega*tau**2/(tau**2*omega**2+1)
    c2=-k*a*omega*tau/(tau**2*omega**2+1)
    c3=k*a*omega/(tau**2*omega**2+1)
    return np.exp(-t/tau)*y_0 + c1/tau*np.exp(-t/tau) \
           + c2*np.cos(omega*t)\
           + c3/omega*np.sin(omega*t)  

def tank2_solution(t,tau1,tau2,k1,k2):
    return k1*k2*(1-(tau1/(tau1-tau2))*np.exp(-t/tau1)-(tau2/(tau2-tau1))*np.exp(-t/tau2))

'''
def impulse(t,kwargs):
    return np.exp(-t/kwargs['tau'])*kwargs['y_0'] \
           + kwargs['k']/kwargs['tau']*np.exp(-t/kwargs['tau'])

def step(t,kwargs):
    return np.exp(-t/kwargs['tau'])*kwargs['y_0'] \
           + kwargs['k']*(1-np.exp(-t/kwargs['tau'])) 
    
def ramp(t,kwargs):
    return np.exp(-t/kwargs['tau'])*kwargs['y_0'] \
           + kwargs['k']*kwargs['tau']*np.exp(-t/kwargs['tau'])\
           + kwargs['k']*(t-kwargs['tau'])

def freq_sine_wave(t,kwargs):
    c1=kwargs['k']*kwargs['a']*kwargs['omega']*kwargs['tau']**2 \
       /(kwargs['tau']**2*kwargs['omega']**2+1)
    c2=-kwargs['k']*kwargs['a']*kwargs['omega']*kwargs['tau'] \
       /(kwargs['tau']**2*kwargs['omega']**2+1)
    c3=kwargs['k']*kwargs['a']*kwargs['omega'] \
       /(kwargs['tau']**2*kwargs['omega']**2+1)
    return np.exp(-t/kwargs['tau'])*kwargs['y_0'] \
           + c1/kwargs['tau']*np.exp(-t/kwargs['tau']) \
           + c2*np.cos(kwargs['omega']*t)+c3/kwargs['omega']*np.sin(kwargs['omega']*t) 
'''