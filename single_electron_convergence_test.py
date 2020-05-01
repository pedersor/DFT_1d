import single_electron, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys

import get_exact_gs_energy


def get_plotting_params():
    # plotting parameters
    params = {'mathtext.default': 'default'}
    plt.rcParams.update(params)
    plt.rcParams['axes.axisbelow'] = True
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 9
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size
    fig, ax = plt.subplots()
    return fig, ax


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


# example checking convergence rate of poschl-teller potential
# for n_point_stencil=5, our slope (p) = 4, for n_point_stencil=3, p = 2.
# TODO(Chris): check convergence for hard wall boundaries using poschl-teller

num_grids_list = [40, 80, 120, 160, 200, 400, 600, 800, 1000]


potential_func_list = [ext_potentials.harmonic_oscillator,
                       functools.partial(ext_potentials.poschl_teller, lam=1),
                       ext_potentials.quartic_oscillator,
                       functools.partial(ext_potentials.gaussian_dips, coeff=1, sigma=1, mu=1),
                       functools.partial(ext_potentials.kronig_penney, a=1, b=0.5, v0=-1),
                       ext_potentials.exp_hydrogenic]

range = (-20, 20)

#exact_energy_grids = np.linspace(-20, 20, 5000)

for potential_fn in potential_func_list:
    
    E_abs_error = []
           
    try:
        func_name = potential_fn.__name__
    except AttributeError:
        func_name = potential_fn.func.__name__


    # obtain lowest eigenvalue (level = 1) from exact/analytical result.
    # When the exact answer is not known, simply run the solver with a large
    # grid, e.g. N = 5000 to obtain the "exact" g.s. energy
    # exact
    # exact_gs_energy = ext_potentials.poschl_teller_eigen_energy(level=1, lam=1)
    exact_gs_energy = get_exact_gs_energy.get_exact_energy_from_npy(potential_fn, range, num_grid = 5000)
    
    print(f'exact gs energy for {func_name}:', exact_gs_energy)
    
    for grid in num_grids_list:
        grids = np.linspace(*range, grid)
        # solve eigenvalue problem with matrix size N = grid
        solver = single_electron.EigenSolver(grids, potential_fn=potential_fn, boundary_condition='closed',
                                             n_point_stencil=5)
    
        
        solver.solve_ground_state()
    
        # obtain lowest eigenvalue from FDM
        ground_state_energy = solver.eigenvalues[0]
    
        # obtain g.s. wavefunction
        ground_state_wf = solver.wave_function[0]
    
        # contruct absolute error
        abs_error = np.abs(ground_state_energy - exact_gs_energy)
        E_abs_error.append(abs_error)
    
    # take (base 10) logs of items in list
    log_ngl = [np.log10(x) for x in num_grids_list]
    log_E = [np.log10(x) for x in E_abs_error]
    
    # skip first 3 small N values for finding linear fit
    log_ngl_fit = log_ngl[3:]
    log_E_fit = log_E[3:]
    
    # linear fitting
    b, p = polyfit(log_ngl_fit, log_E_fit, 1)
    r2 = '%.4f' % (rsquared(log_ngl_fit, log_E_fit))
    yfit = [10 ** (b + p * xi) for xi in log_ngl]
    p = '%.4f' % (p)
    
    size_diff = np.abs(len(num_grids_list) - len(yfit))
    
    # initialize figure for plots
    fig, ax = get_plotting_params()
    # obtain linear fit of data (skipping first 3 small N values).
    # here p = slope in our fit, r^2 is a measure of how linear data is.
    linfit = ax.plot(num_grids_list[size_diff:], yfit, alpha=0.4,
                     label='$p$ = ' + p + ', $r^2$ = ' + r2, linewidth=3)
    # matplotlib trick to obtain same color of a previous plot
    ax.plot(num_grids_list, E_abs_error, marker='o', linestyle='None',
            color=linfit[0].get_color())
    
    # log-log scale
    plt.xscale('log')
    plt.yscale('log')
    
    ax.set_xlabel("$N$", fontsize=18)
    ax.set_ylabel("|Error| (au)", fontsize=18)
    
    plt.legend(fontsize=16)
    plt.title(f'Error in ground state vs. number of grids\n{func_name}', fontsize=20)
    plt.grid(alpha=0.4)
    plt.gca().xaxis.grid(True, which='minor', alpha=0.4)
    plt.gca().yaxis.grid(True, which='minor', alpha=0.4)
    
    plt.savefig(f'ground_states_npy/{func_name}.png')
    plt.close()
    
    
