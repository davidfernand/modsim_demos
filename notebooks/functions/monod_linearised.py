"""
Functions for the linearised version of the biochemical reactor with Monod kinetics

David Fernandes del Pozo
"""
# ----------------------------
# Implementation biochemical reactor with Monod kinetics
# ----------------------------

def model_derivatives_linearised(variables, t, kwargs):
    '''
    $$\frac{dX}{dt} \approx \Big(\mu_{max}\frac{\bar{S}_{in}}{K_{S}+\bar{S}_{in}}X - \frac{Q}{V}\Big)X$$
    $$\frac{dS}{dt} \approx -\frac{1}{Y} \mu_{max}\frac{\bar{S}_{in}}{K_{S}+\bar{S}_{in}}X-\frac{Q}{V}(S-\bar{S}_{in})$$
    '''
    X = variables[0]
    S = variables[1]

    dXdt = (kwargs['mu_max']*(kwargs['S_in']/(kwargs['K_S']+kwargs['S_in']))\
                  -kwargs['Q']/kwargs['V'])*X
       
    dSdt = -(kwargs['mu_max']*kwargs['S_in']/(kwargs['Y']*(kwargs['K_S']+kwargs['S_in'])))*X \
            -kwargs['Q']/kwargs['V']*(S-kwargs['S_in'])

    return [dXdt, dSdt]

'''
from scipy import integrate
def solve_Monod_lin_model(t,u,mu_max,Y,Ks,Q,V,S_in,init):
    def model(u,t):

        #Differential equation

        dXdt = (mu_max*(S_in/(Ks+S_in))-Q/V)*u[0]#+Ks*mu_max/(Ks+S_in)**2*u[0]*(u[1]-S_in)  #Biomass X
        dSdt = -(mu_max*S_in/(Y*(Ks+S_in)))*u[0]-Q/V*(u[1]-S_in) #- Ks*mu_max/(Y*(Ks+S_in)**2)*u[0]*(u[1]-S_in)  #Reactant S

        return [dXdt,dSdt]
    y=odeint(model,init,time)
    return y'''
