"""
Function for a simple Monod equation

David Fernandes del Pozo
"""
# ----------------------------
# Implementation of the Monod model
# ----------------------------

def Monod_kinetics(S,mu_max,K_s):
    '''
    $$\mu = \mu_{max}\cdot \frac{S}{K_s+S}$$
    '''
    return  mu_max*S/(K_s+S)
