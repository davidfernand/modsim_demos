"""
Functions for the biochemical reactor with Monod kinetics

Daan Van Hauwermeiren, David Fernandes del Pozo
"""
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

    dXdt = kwargs['mu_max']*S/(kwargs['K_S'] + S)*X \
          -kwargs['Q']/kwargs['V']*X
    dSdt = -kwargs['mu_max']/kwargs['Y']*S/(kwargs['K_S']+S)*X \
        +kwargs['Q']/kwargs['V']*(kwargs['S_in']-S)
    return [dXdt, dSdt]
