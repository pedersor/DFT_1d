"""
.. _single_electron_test:

Test for Single Electron Module
###############################

.. todo::

    * Authors? -RJM
    * Docs need love
    * Should validate correct instiliation/completion. Right now just spits printouts. -RJM
    * Ideally a single test script would test EVERY module, and can be easily run after each git commit. May need to make a another test script which calls this one and all others. -RJM
    * Has this been linted yet? -RJM

"""

import sys
import os
currentpath = os.path.abspath('.')
sys.path.insert(0, os.path.dirname(currentpath))

import single_electron, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import time
import warnings


def get_plotting_params():
    """ Convergence plotting parameters. """
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


def convergence_test(Solver,
                     test_range,
                     potential_fn,
                     boundary_condition,
                     n_point_stencil,
                     k_point=None,
                     num_grids_list=None,
                     analytical_energy=None,
                     plot_index=''):
    """Description.

    .. todo::

        * Please fill out docs

    Args:
      ...

    Returns:
      ...
    """
    # start timer
    t0 = time.time()

    if num_grids_list is None:
        num_grids_list = [40, 80, 120, 160, 200, 400, 600, 800, 1000]

    # error list for plotting
    E_abs_error = []

    # get the name of potential function in order to save to local machine
    try:
        func_name = potential_fn.__name__
    except AttributeError:
        func_name = potential_fn.func.__name__

    # choose whether include endpoints
    if boundary_condition == 'periodic':
        endpoint = False
    else:
        endpoint = True

    # obtain lowest eigenvalue (level = 1) from exact/analytical result.
    # When the exact answer is not known, simply run the solver with a large
    # grid, e.g. N = 5000 to obtain the "exact" g.s. energy
    if analytical_energy:
        exact_gs_energy = analytical_energy
        energy_form = 'analytical'
    else:
        # solve eigenvalue problem with matrix size N = 5000
        exact_grids = np.linspace(*test_range, 5000, endpoint=endpoint)
        exact_solver = Solver(exact_grids,
                              potential_fn=potential_fn,
                              k_point=k_point,
                              boundary_condition=boundary_condition,
                              n_point_stencil=n_point_stencil)

        # solve ground state
        exact_solver.solve_ground_state()

        # obtain ground state energy as exact energy
        exact_gs_energy = exact_solver.eigenvalues[0]
        energy_form = '5000_grids'

    # get error of energy for each num_grid compared to the exact energy
    for num_grids in num_grids_list:
        grids = np.linspace(*test_range, num_grids, endpoint=endpoint)
        # solve eigenvalue problem with matrix size N = num_grids
        solver = Solver(grids,
                        potential_fn=potential_fn,
                        k_point=k_point,
                        boundary_condition=boundary_condition,
                        n_point_stencil=n_point_stencil)

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
    plt.title(f'Error in ground state vs. number of grids\n{func_name}, '
              f'{boundary_condition}, {test_range}, {n_point_stencil}-points, '
              f'{energy_form}',
              fontsize=20)
    plt.grid(alpha=0.4)
    plt.gca().xaxis.grid(True, which='minor', alpha=0.4)
    plt.gca().yaxis.grid(True, which='minor', alpha=0.4)

    # create folder if no such directory
    if not os.path.isdir('convergence_test'):
        os.mkdir('convergence_test')
    if not os.path.isdir(f'convergence_test/{Solver.__name__}'):
        os.mkdir(f'convergence_test/{Solver.__name__}')

    # save fig
    plt.savefig(f'convergence_test/{Solver.__name__}/{func_name}_'
                f'{boundary_condition}_{test_range}_{n_point_stencil}_'
                f'{energy_form}{plot_index}.png')
    plt.close()

    # stop timer
    t1 = time.time()

    # write time taken to complete the convergence test to log (txt) file
    time_str = time.strftime("==== %Y-%m-%d %H:%M:%S ====", time.localtime())
    finish_str = f'{Solver.__name__}: {func_name}_{boundary_condition}_' \
                 f'{test_range}_{n_point_stencil}_{energy_form}{plot_index}'
    timer_str = f'Time: {t1 - t0}'
    all_str = time_str + '\n' + finish_str + '\n' + timer_str + '\n\n'

    with open("convergence_test/test_log.txt", "a") as text_file:
        text_file.write(all_str)

    print(all_str)


# plot the dispersion relation for a periodic potential
# TODO: move this to an example (not a test)
def plot_dispersion(Solver,
                    test_range,
                    potential_fn,
                    k_range=(-np.pi, np.pi),
                    eigenvalue_index=0,
                    n_point_stencil=5,
                    num_grids=1000,
                    num_k_grids=100):
    warnings.warn('Warning: make sure potential_fn is a periodic function!')

    # grids = np.linspace(*test_range, num_grids, endpoint = False)
    k_list = np.linspace(*k_range, num_k_grids)
    E_list = []

    for k in k_list:
        grids = np.linspace(*test_range, num_grids, endpoint=False)
        solver = Solver(grids,
                        potential_fn=potential_fn,
                        k_point=k,
                        boundary_condition='periodic',
                        n_point_stencil=n_point_stencil,
                        tol=0)

        solver.solve_ground_state()

        # obtain lowest eigenvalue from FDM
        energy = solver.eigenvalues[eigenvalue_index]

        E_list.append(energy)

    # initialize figure for plots
    fig, ax = get_plotting_params()
    # matplotlib trick to obtain same color of a previous plot
    ax.plot(k_list, E_list, marker='o', linestyle='solid', color='blue')

    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel("E", fontsize=18)

    # plt.legend(fontsize=16)
    # plt.title(f'Dispersion relation {k_range} {eigenvalue_index}', fontsize=20)
    plt.grid(alpha=0.4)
    plt.gca().xaxis.grid(True, which='minor', alpha=0.4)
    plt.gca().yaxis.grid(True, which='minor', alpha=0.4)

    # create folder if no such directory
    if not os.path.isdir('dispersion_plots'):
        os.mkdir('dispersion_plots')

    # save fig
    plt.savefig(f'dispersion_plots/dispersion_relation_{k_range}_'
                f'{eigenvalue_index}.png')
    plt.close()

    print(f'dispersion_relation_{k_range}_{eigenvalue_index} done')


if __name__ == "__main__":
    """ Test convergence rates for various systems."""

    test_potential_fn_list = [((0, 3),
                               functools.partial(ext_potentials.kronig_penney,
                                                 a=3, b=0.5, v0=-1),
                               'periodic'),
                              ((-5, 5),
                               functools.partial(ext_potentials.poschl_teller,
                                                 lam=1), 'closed'),
                              ((-20, 20),
                               functools.partial(ext_potentials.poschl_teller,
                                                 lam=1), 'open'),
                              ((0, 2 * np.pi), np.sin, 'periodic')]

    solvers = [single_electron.SparseEigenSolver, single_electron.EigenSolver]

    # convergence test for the sin periodic potential on arbitrary k_point
    r, p, b = test_potential_fn_list[3]
    convergence_test(solvers[0], r, p, b, 5, k_point=1)
