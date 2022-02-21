"""
General functions

Daan Van Hauwermeiren, David Fernandes del Pozo
"""
# Import all relevant python functions and libraries
import matplotlib.pyplot as plt
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
    "lines.linewidth": 4,#1.75,
    "patch.linewidth": .3,
    "lines.markersize": 7,
    "lines.markeredgewidth": 0,

    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": .5,
    "ytick.minor.width": .5,

    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
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
        fig, ax = plt.subplots(figsize=figsize)
        if twinax: # Only for the heat balance example (hardcoded)
            modeloutput['$N_2O_5$'].plot(ax=ax);
            modeloutput['$N_2O_4$'].plot(ax=ax);
            ax_twin = ax.twinx();
            ax_twin.yaxis.label.set_color('blue');
            ax_twin.tick_params(axis='y', colors='blue');
            modeloutput['$\mathrm{T}$'].plot(ax=ax_twin,color='blue');
        else:
            modeloutput.plot(ax=ax);
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

    perturbation: float
        perturbation of the parameter

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
    fig, ax = plt.subplots(figsize=figsize)
    sens.plot(ax=ax)
    ax.set_xlabel(r'$\mathrm{Time}$')
    ax.set_ylabel(sort)

def sse(simulation, data):
    return np.sum(np.sum((np.atleast_2d(data) - np.atleast_2d(simulation))**2))

def track_calib(opt_fun, X, param_names, method='Nelder-Mead', tol=1e-4):
    """
    Optimisation using the Nelder-Mead algorithm. All iteration steps are tracked.

    Arguments
    ----------
    opt_fun : function
        optimisation function
    X : list
        parameters
    method : str
        Define the method for optimisation, options are: 'Nelder-Mead', 'BFGS', 'Powell'
        basinhopping', 'brute', 'differential evolution'
    toll: float
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

    if method in ['Nelder-Mead', 'BFGS', 'Powell']:
        res = optimize.minimize(internal_opt_fun, X, method=method, tol=tol)
    elif method == 'basinhopping':
        res = optimize.basinhopping(internal_opt_fun, X)
    elif method == 'brute':
        bounds = [(0.01*i, 10*i) for i in X]
        res = optimize.brute(internal_opt_fun, bounds)
    elif method == 'differential evolution':
        bounds = [(0.01*i, 100*i) for i in X]
        res = optimize.differential_evolution(internal_opt_fun, bounds, tol=tol)
    else:
        raise ValueError('use correct optimisation algorithm, see docstring for options')
    parameters = pd.DataFrame(np.array(parameters), columns=param_names)
    results = np.array(results)

    return parameters,results

def plot_calib(parameters, results, i, data, sim_model):
    fig, ax = plt.subplots(figsize=figsize)
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
    ax.legend(handles, labels, loc='best')
    fig, ax = plt.subplots(figsize=figsize)
    cols = parameters.columns
    c = results - min(results)
    c *= 1/max(c)
    sc = ax.scatter(parameters[cols[0]], parameters[cols[1]], c=c, s=50, cmap="viridis", vmax=1)
    cbar = plt.colorbar(sc)
    cbar.set_ticks([0.05*max(c), 0.95*max(c)])
    cbar.set_ticklabels(['Low value \nobjective function', 'High value\nobjective function'])
    ax.scatter(parameters[cols[0]].iloc[0], parameters[cols[1]].iloc[0], marker='o', s=450, c=fivethirtyeight[2]) # startwaarde
    ax.scatter(parameters[cols[0]].iloc[-1], parameters[cols[1]].iloc[-1], marker='*', s=500, c=fivethirtyeight[1]) # eindwaarde
    ax.scatter(parameters[cols[0]].iloc[i], parameters[cols[1]].iloc[i], s=150, vmax=1, c=fivethirtyeight[4])        # huidige waarde
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_xlim(0.95*parameters[cols[0]].min(), 1.05*parameters[cols[0]].max())
    ax.set_ylim(0.95*parameters[cols[1]].min(), 1.05*parameters[cols[1]].max())

def plot_contour_monod(optimizer):
    n_points = 30
    #n_points = kwargs[4]
    mu_max = np.logspace(np.log10(0.001), np.log10(50), n_points)
    #mu_max = np.logspace(np.log10(kwargs[0]), np.log10(kwargs[1]), n_points)
    K_S = np.logspace(np.log10(0.001), np.log10(50), n_points)
    #K_S = np.logspace(np.log10(kwargs[2]), np.log10(kwargs[3]), n_points)
    X_mu_max, X_K_S = np.meshgrid(mu_max, K_S)
    Z = np.array([optimizer(params) for params in zip(X_mu_max.flatten(), X_K_S.flatten())])
    Z = Z.reshape((n_points, n_points))
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.contourf(X_mu_max, X_K_S, Z, cmap='viridis')
    cbar = plt.colorbar(sc)
    cbar.set_ticks([0.05*Z.max(), 0.95*Z.max()])
    cbar.set_ticklabels(['Low value \nobjective function', 'High value\nobjective function'])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel('mu_max')
    ax.set_ylabel('K_S')

def plot_contour_force(optimizer):
    n_points = 30
    b = np.linspace(0, 2, n_points)
    k = np.linspace(0, 2, n_points)
    X_b, X_k = np.meshgrid(b, k)
    Z = np.array([optimizer(params) for params in zip(X_b.flatten(), X_k.flatten())])
    Z = np.log10(Z)
    Z = Z.reshape((n_points, n_points))
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.contourf(X_b, X_k, Z, cmap='viridis')
    cbar = plt.colorbar(sc)
    cbar.set_ticks([0.05*Z.max(), 0.95*Z.max()])
    cbar.set_ticklabels(['Low value \nobjective function', 'High value\nobjective function'])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel('b')
    ax.set_ylabel('k')
