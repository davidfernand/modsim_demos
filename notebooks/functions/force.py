"""
Functions for the force balance model

Ingmar Nopens, Daan Van Hauwermeiren, David Fernandes del Pozo
"""

# ----------------------------
# Implementation of the force model with the harmonic oscillator
# ----------------------------

def model_derivatives(variables, t, kwargs):

    Position = variables[0]
    Velocity = variables[1]

    dPositiondt = Velocity
    dVelocitydt = -kwargs['b']/kwargs['m']*Velocity \
                  - kwargs['k']/kwargs['m']*Position \
                  + kwargs['Fex']
    return [dPositiondt, dVelocitydt]
