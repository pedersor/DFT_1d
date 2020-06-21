import single_electron, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys
import os

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
def convergence_test(Solver,
                     range,
                     potential_fn,
                     boundary_condition,   
                     n_point_stencil,  
                     num_grids_list = [40, 80, 120, 160, 200, 400, 600, 800, 1000],
                     analytical_energy = None):
    
    # error list for plotting
    E_abs_error = []
    
    # get the name of potential function in order to save to local machine
    try:
        func_name = potential_fn.__name__
    except AttributeError:
        func_name = potential_fn.func.__name__


    # obtain lowest eigenvalue (level = 1) from exact/analytical result.
    # When the exact answer is not known, simply run the solver with a large
    # grid, e.g. N = 5000 to obtain the "exact" g.s. energy
    if analytical_energy:
        exact_gs_energy = analytical_energy
        energy_form = 'analytical'
    else:
        # solve eigenvalue problem with matrix size N = 5000
        exact_grids = np.linspace(*range, 5000)      
        exact_solver = Solver(exact_grids,
                              potential_fn = potential_fn, 
                              boundary_condition = boundary_condition,
                              n_point_stencil = n_point_stencil)
        
        # solve ground state
        exact_solver.solve_ground_state()
        
        # obtain ground state energy as exact energy
        exact_gs_energy = exact_solver.eigenvalues[0]
        energy_form = '5000_grids'
    
    for grid in num_grids_list:
        grids = np.linspace(*range, grid)
        # solve eigenvalue problem with matrix size N = grid
        solver = Solver(grids, 
                        potential_fn = potential_fn, 
                        boundary_condition = boundary_condition,
                        n_point_stencil = n_point_stencil)
    
        
        solver.solve_ground_state()
    
        # obtain lowest eigenvalue from FDM
        ground_state_energy = solver.eigenvalues[0]
    
        # obtain g.s. wavefunction
        # ground_state_wf = solver.wave_function[0]
    
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
    plt.title(f'Error in ground state vs. number of grids\n{func_name}, {boundary_condition}, {range}, {n_point_stencil}-points, {energy_form}', fontsize=20)
    plt.grid(alpha=0.4)
    plt.gca().xaxis.grid(True, which='minor', alpha=0.4)
    plt.gca().yaxis.grid(True, which='minor', alpha=0.4)
    
    # create folder if no such directory
    if not os.path.isdir('convergence_test'):
        os.mkdir('convergence_test')
    
    # save fig
    plt.savefig(f'convergence_test/{func_name}_{boundary_condition}_{range}_{n_point_stencil}_{energy_form}.png')
    plt.close()
    
    # message for completing one convergence test
    print(f'{func_name}_{boundary_condition}_{range}_{n_point_stencil}_{energy_form} done')
        


if __name__ == "__main__":
    
    num_grids_list = [40, 80, 120, 160, 200, 400, 600, 800, 1000]
    test_potential_fn_list = [functools.partial(ext_potentials.poschl_teller, lam=1), 
                              ext_potentials.exp_hydrogenic]
    
    Solver = single_electron.EigenSolver
    
    for potential_fn in test_potential_fn_list:
        convergence_test(Solver, (-5, 5), potential_fn, 'closed', 3)
        convergence_test(Solver, (-5, 5), potential_fn, 'closed', 5)
        convergence_test(Solver, (-20, 20), potential_fn, 'closed', 3)
        convergence_test(Solver, (-20, 20), potential_fn, 'closed', 5)


