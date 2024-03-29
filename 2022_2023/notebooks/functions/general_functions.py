"""
General functions to be used for the notebooks

Authors: Daan Van Hauwermeiren, David Fernandes del Pozo
"""
# Import all relevant python functions and libraries
from cProfile import label
import matplotlib as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import math

# For solver
from scipy.integrate import odeint
from scipy import optimize
from IPython.display import display

# Plot settings
base_context = {

    "font.size": 12,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    "grid.linewidth": 1,
    "lines.linewidth": 3,#1.75,
    "patch.linewidth": .3,
    "lines.markersize": 7,
    "lines.markeredgewidth": 0,

    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": .5,
    "ytick.minor.width": .5,

    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
    "axes.grid":    True,
    "text.usetex":False,
    "mathtext.fontset": "cm"
    }

context = 'notebook'
font_scale = 2

# Scale all the parameters by the same factor depending on the context
scaling = dict(paper=.8, notebook=1, talk=1.3, poster=1.6)[context]
context_dict = {k: v * scaling for k, v in base_context.items()}

# Now independently scale the fonts
font_keys = ["axes.labelsize", "axes.titlesize", "legend.fontsize",
             "xtick.labelsize", "ytick.labelsize", "font.size"]
font_dict = {k: context_dict[k] * font_scale for k in font_keys}
context_dict.update(font_dict)

plt.rcParams.update(context_dict)

# Set up some colors to differentiate between the variables
fivethirtyeight = ["black", "#fc4f30", "blue", "#6d904f", "#8b8b8b"]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', fivethirtyeight)

figsize = (9,6)

def model(timesteps, init, varnames, f, returnDataFrame=False,
          plotresults=True,twinax=False, **kwargs):
    """
    Model implementation

    Parameters
    -----------
    time steps: np.array
        array of time steps

    init: list
        list of initial conditions

    varnames: list
        list of strings with names of the variables

    f: function
        function that defines the derivatives to be solved

    returnDataFrame: bool
        set to True to get back the simulation data

    twinax: bool
        set to True to plot temperature to secondary axis in heat balance case

    kwargs: dict
        function specific parameters

    """
    fvals = odeint(f, init, timesteps, args=(kwargs,))
    data = {col:vals for (col, vals) in zip(varnames, fvals.T)}
    idx = pd.Index(data=timesteps, name=r'$\mathrm{Time}$')
    modeloutput = pd.DataFrame(data, index=idx)
    
    if plotresults:
        fig, ax = plt.pyplot.subplots(figsize=figsize)
        if twinax: # Only for the heat balance example (hardcoded)
            modeloutput['$N_2O_5$'].plot(ax=ax);
            modeloutput['$N_2O_4$'].plot(ax=ax);
            ax_twin = ax.twinx();
            ax_twin.yaxis.label.set_color('blue');
            ax_twin.tick_params(axis='y', colors='blue');
            modeloutput[r'$\mathrm{T}$'].plot(ax=ax_twin,color='blue');
            ax_twin.grid(False)
            ax.grid(False)
        else:
            font = font_manager.FontProperties(family='serif',style='normal', size=16)
            labels =  ['{}'.format(x) for x in varnames]
            modeloutput.plot(ax=ax,label=labels);
            ax.legend(loc='best',prop=font,framealpha=1);
    if returnDataFrame:
        return modeloutput;


def sensitivity(timesteps, init, varnames, f, parametername,
                  log_perturbation=-4, sort='absolute', **kwargs):
    """
    Calculates the sensitivity function(s) of the model output(s) to 1 given parameter

    Arguments
    -----------
    timesteps: np.array
        array of timesteps

    init: list
        list of initial conditions

    varnames: list
        list of strings with names of the variables

    f: function
        function that defines the derivatives to be solved

    parametername: string
        name of the parameter for which the sensitivity function is to be set up.

    log_perturbation: float
        Exponent of the logarithmic perturbation in base 10 of the parameter

    kwargs: dict
        function specific parameters

    """
    perturbation = 10**log_perturbation
    res_basis = model(timesteps, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    parametervalues = kwargs.pop(parametername)
    kwargs[parametername] = (1 + perturbation) * parametervalues
    res_high = model(timesteps, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    kwargs[parametername] = (1 - perturbation) * parametervalues
    res_low = model(timesteps, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    if sort == 'absolute sensitivity':
        sens = (res_high - res_low)/(2.*perturbation)

    if sort == 'relative sensitivity parameter':
            sens = (res_high - res_low)/(2.*perturbation)*parametervalues

    if sort == 'relative sensitivity variable':
        sens = (res_high - res_low)/(2.*perturbation)/res_basis

    if sort == 'relative total sensitivity':
        sens = (res_high - res_low)/(2.*perturbation)*parametervalues/res_basis
    fig, ax = plt.pyplot.subplots(figsize=figsize)
    sens.plot(ax=ax)
    ax.set_xlabel(r'$\mathrm{Time}$',size=20)
    font_label = {'family':'serif','size':20}
    ax.set_ylabel(sort,fontdict=font_label)
    font_legend = font_manager.FontProperties(family='serif',style='normal', size=20)
    ax.legend(loc='best',prop=font_legend,framealpha=1);

def sse(simulation, data):
    return np.sum(np.sum((np.atleast_2d(data) - np.atleast_2d(simulation))**2))

def track_calib(opt_fun, X, param_names, method='Nelder-Mead',bounds='none',maxiter='none', tol='none'):
    """
    Optimisation using a supplied minimisation algorithm. All iteration steps are tracked.

    Arguments
    ----------
    opt_fun : function
        optimisation function
    X : list
        parameters
    method : str
        Define the method for optimisation, options are: 'Nelder-Mead' (default), 'BFGS', 'Powell'
        'L-BFGS-B','basinhopping', 'brute', 'differential evolution'
    bounds:  tuple
        Provides bounds for the optimisation algorithm (only for Nelder-Mead and BFGS, e.g. ((0,1),(0,1)))
    maxiter: int
        Sets a maximum number of minimization iterations (only for Nelder-Mead and BFGS)
    tol: float
        tolerance to determine the endpoint of the optimisation. Is not used in
        method options 'brute' and 'basinhopping
    Output
    ------
    parameters : DataFrame
        all tested parameter combinations
    results : np.array
        all obtained objective values

    """
    parameters = []
    results = []

    def internal_opt_fun(X):
        result = opt_fun(X) # Calculate object function value for current set of parameters
        parameters.append(X) # Keep track of intermediate parameter values
        results.append(result) # Keep track of intermediate SSE
        return result

    if method in ['Nelder-Mead', 'Powell','L-BFGS-B']:
        res = optimize.minimize(internal_opt_fun, X, method=method,
        bounds=bounds,options  = {"maxiter":maxiter}, tol=tol);
    elif method=='BFGS':
        res = optimize.minimize(internal_opt_fun, X, method=method,
        options  = {"maxiter":maxiter}, tol=tol);
    elif method=='Newton-CG':
        fprime = lambda x: optimize.approx_fprime(x, internal_opt_fun, 0.001)
        res = optimize.minimize(internal_opt_fun, X, method=method,
        jac=fprime,options  = {"maxiter":maxiter}, tol=tol);
    elif method == 'basinhopping':
        res = optimize.basinhopping(internal_opt_fun, X);
    elif method == 'brute':
        bounds = [(0.01*i, 10*i) for i in X]
        res = optimize.brute(internal_opt_fun, bounds);
    elif method == 'differential evolution':
        bounds = [(0.01*i, 100*i) for i in X]
        res = optimize.differential_evolution(internal_opt_fun, bounds, tol=tol);
    else:
        raise ValueError('Use correct optimisation algorithm, see docstring for options')

    parameters = pd.DataFrame(np.array(parameters), columns=param_names)
    results = np.array(results)

    return parameters,results

def plot_calib(parameters, results, i, data, sim_model):
    fig, ax = plt.pyplot.subplots(figsize=figsize)
    cols = data.columns
    data[cols].plot(ax=ax, linestyle='', marker='.', markersize=15,
              color=[fivethirtyeight[0], fivethirtyeight[1]])
    sim = sim_model(parameters.loc[i].values)
    sim[cols].plot(ax=ax, linewidth=5,
             color=[fivethirtyeight[0], fivethirtyeight[1]])
    ax.set_xlabel(r'$\mathrm{Time}$')
    ax.set_ylabel('Variable values');
    handles, labels = ax.get_legend_handles_labels()
    labels = [l+' simulation' if (i>= data.shape[1]) else l for i, l in enumerate(labels)]
    font_legend = font_manager.FontProperties(family='serif',style='normal', size=20)
    ax.legend(handles, labels,prop=font_legend, loc='best',framealpha=1)
    fig, ax = plt.pyplot.subplots(figsize=figsize)
    cols = parameters.columns
    c = results - min(results)
    c *= 1/max(results)
    sc = ax.scatter(parameters[cols[0]], parameters[cols[1]], c=c, s=50, cmap="viridis", vmax=1,label='_nolegend_')
    cbar = plt.pyplot.colorbar(sc, format='%.2f')
    #cbar.set_ticks([0.05*max(c), 0.95*max(c)])
    cbar.set_ticks([min(c), max(c)])
    #cbar.set_ticklabels([np.round(min(results),3), np.round(max(results),3)])
    cbar.set_label(r'$\frac{J(\theta)-J(\hat{\theta})}{max(J(\theta))}$', rotation=0,fontsize=30)
    #cbar.set_ticklabels(['Low value \nobjective function', 'High value\nobjective function'])
    ax.scatter(parameters[cols[0]].iloc[0], parameters[cols[1]].iloc[0], marker='o', s=300,c='b',label=r'$\mathrm{Initial}$') 
    ax.scatter(parameters[cols[0]].iloc[-1], parameters[cols[1]].iloc[-1], marker='*', s=300, c=fivethirtyeight[1],label=r'$\mathrm{Optimal}$') 
    ax.scatter(parameters[cols[0]].iloc[i], parameters[cols[1]].iloc[i], marker='o', s=200, vmax=1,facecolors='none', edgecolors='k',linewidths=2,label=r'$\mathrm{Solver}$')
    ax.legend(loc='best')
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_xlim(0.95*parameters[cols[0]].min(), 1.05*parameters[cols[0]].max())
    ax.set_ylim(0.95*parameters[cols[1]].min(), 1.05*parameters[cols[1]].max())

def plot_contour(optimizer,optimal, **kwargs):

    if kwargs['model'] == 'monod':
        n_points = kwargs['n_points']
        mu_max_opt = optimal[0]
        K_s_opt = optimal[1]
        #n_points = kwargs[4]
        mu_max = np.logspace(np.log10(kwargs['mumax'][0]), np.log10(kwargs['mumax'][1]), n_points)
        K_S = np.logspace(np.log10(kwargs['Ks'][0]), np.log10(kwargs['Ks'][1]), n_points)
        X_mu_max, X_K_S = np.meshgrid(mu_max, K_S)
        Z = np.array([optimizer(params) for params in zip(X_mu_max.flatten(), X_K_S.flatten())])
        Z = Z.reshape((n_points, n_points))
        fig, ax = plt.pyplot.subplots(figsize=(6.5,5.5))
        sc = ax.contourf(X_mu_max, X_K_S, Z, cmap='viridis')
        ax.scatter(mu_max_opt,K_s_opt, marker='*', s=500, c=fivethirtyeight[1]) 

        cbar = plt.pyplot.colorbar(sc, format='%.2e')
        cbar.set_ticks([0.05*Z.max(), 0.95*Z.max()])
        cbar.set_label(r'$J(\theta)$', rotation=0)
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_xlabel(r'$\mu_{max}$')
        ax.set_ylabel(r'$K_S$')

    elif kwargs['model'] == 'force':

        n_points = kwargs['n_points']
        mu_max_opt = optimal[0]
        K_s_opt = optimal[1]
        #n_points = kwargs[4]
        mu_max = np.logspace(np.log10(kwargs['b'][0]), np.log10(kwargs['b'][1]), n_points)
        K_S = np.logspace(np.log10(kwargs['k'][0]), np.log10(kwargs['k'][1]), n_points)
        X_mu_max, X_K_S = np.meshgrid(mu_max, K_S)
        Z = np.array([optimizer(params) for params in zip(X_mu_max.flatten(), X_K_S.flatten())])
        Z = Z.reshape((n_points, n_points))
        fig, ax = plt.pyplot.subplots(figsize=(6.5,5.5))
        sc = ax.contourf(X_mu_max, X_K_S, Z, cmap='viridis')
        ax.scatter(mu_max_opt,K_s_opt, marker='*', s=500, c=fivethirtyeight[1]) 

        cbar = plt.pyplot.colorbar(sc, format='%.2e')
        cbar.set_ticks([0.05*Z.max(), 0.95*Z.max()])
        cbar.set_label(r'$J(\theta)$', rotation=0)
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_xlabel(r'$b$')
        ax.set_ylabel(r'$k$')
    
    else:
        raise ValueError('Use model = monod or force to plot the contours')
