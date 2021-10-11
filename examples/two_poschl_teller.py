"""
.. _two_poschl_teller:

Poschl-Teller potentials
########################

Summary:
    Calculates a system with two Poschl-Teller wells
"""

import sys
import os

import numpy as np
import functools
import matplotlib.pyplot as plt

from DFT_1d.ext_potentials import poschl_teller
from DFT_1d.non_interacting_solver import SparseEigenSolver


def get_plotting_params():
    r'''Initialize figure for plots.
    
    Returns:
      fig: a Figure object with initialized parameters.
      ax: an Axes object with initialized parameters.
    '''

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


def plot_and_save(x_list, y_list, x_label, y_label, name,
                  xlim=(None, None), ylim=(None, None), folder=None):
    r'''Plot a single curve on a graph and save it as .png file.
    
    Args:
      x_list: list (or array), specify the x-axis points.
      y_list: list (or array), specify the y-axis points.
      x_label: label for x-axis.
      y_label: label for y-axis.
      name: str, name of the image saved (without .png).
      folder: str, the path to the folder to save the image in. Does not need
          to pre-exist before running.
    '''

    # initialize figure for plots
    fig, ax = get_plotting_params()

    # matplotlib trick to obtain same color of a previous plot
    ax.plot(x_list, y_list, marker='o', linestyle='solid', color='blue')

    # set labels for x and y axis
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)

    # set x and y limits
    xlim_l = xlim[0]
    xlim_r = xlim[1]
    ylim_b = ylim[0]
    ylim_t = ylim[1]
    if xlim_l != None:
        ax.set_xlim(left=xlim_l)
    if xlim_r != None:
        ax.set_xlim(right=xlim_r)
    if ylim_b != None:
        ax.set_ylim(bottom=ylim_b)
    if ylim_t != None:
        ax.set_ylim(top=ylim_t)

    # set grids
    plt.grid(alpha=0.4)
    plt.gca().xaxis.grid(True, which='minor', alpha=0.4)
    plt.gca().yaxis.grid(True, which='minor', alpha=0.4)

    # save image
    if folder != None:
        # create folder if no such directory
        if not os.path.isdir(folder):
            os.mkdir(folder)
        # save fig
        plt.savefig(f'{folder}/{name}.png')
        plt.close()
    else:
        # save fig
        plt.savefig(f'{name}.png')
        plt.close()


def plot_multiple_and_save(plot_list, name,
                           xlim=(None, None), ylim=(None, None), folder=None):
    r'''Plot mutiple curves on a single graph and save it as .png file.
    
    Args:
      plot_list: list of lists, specify plotting parameters for each curve
          in the form of [label, x_list, y_list, linestyle, color].
      name: str, name of the image saved (without .png).
      xlim: tuple, xlim[0] == left limit, xlim[1] == right limit; setting to
          None means flexible ranges.
      ylim: tuple, ylim[0] == bottom limit, ylim[1] == top limit; setting to
          None means flexible ranges.
      folder: str, the path to the folder to save the image in. Does not need
          to pre-exist before running.
    '''

    # initialize figure for plots
    fig, ax = get_plotting_params()

    # matplotlib trick to obtain same color of a previous plot
    for data_list in plot_list:
        label = data_list[0]
        x_list = data_list[1]
        y_list = data_list[2]
        linestyle = data_list[3]
        color = data_list[4]
        ax.plot(x_list, y_list, label=label, linestyle=linestyle, color=color)

    # set x label
    ax.set_xlabel('x', fontsize=18)

    # set x and y limits
    xlim_l = xlim[0]
    xlim_r = xlim[1]
    ylim_b = ylim[0]
    ylim_t = ylim[1]
    if xlim_l != None:
        ax.set_xlim(left=xlim_l)
    if xlim_r != None:
        ax.set_xlim(right=xlim_r)
    if ylim_b != None:
        ax.set_ylim(bottom=ylim_b)
    if ylim_t != None:
        ax.set_ylim(top=ylim_t)

    # set grids
    plt.grid(alpha=0.4)
    plt.gca().xaxis.grid(True, which='minor', alpha=0.4)
    plt.gca().yaxis.grid(True, which='minor', alpha=0.4)

    # set legend
    plt.legend(loc=2)

    # save image
    if folder != None:
        # create folder if no such directory
        if not os.path.isdir(folder):
            os.mkdir(folder)
        # save fig
        plt.savefig(f'{folder}/{name}.png')
        plt.close()
    else:
        # save fig
        plt.savefig(f'{name}.png')
        plt.close()


def two_poschl_teller(grids, d, lam=1., a=1.):
    r"""Two Poschl-Teller potential wells seperated by distance d.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      d: float, distance seperated by two Poschl-Teller potential wells.
      lam: float, lambda in the Poschl-Teller potential function.
      a: float, coefficient in the Poschl-Teller potential function.
      center: float, the center of the potential.

    Returns:
      Potential on grid with shape (num_grid,)

    Raises:
      ValueError: If lam is not positive.
    """

    potential_grids = poschl_teller(grids, lam=lam, a=a, center=-d / 2)
    potential_grids += poschl_teller(grids, lam=lam, a=a, center=d / 2)
    return potential_grids


if __name__ == '__main__':

    # initialize variables
    test_range = (-20, 20)
    grids = np.linspace(*test_range, 1000)
    boundary_condition = 'open'
    n_point_stencil = 5
    d_list = []
    E_list = []
    solver = SparseEigenSolver(grids,
                               boundary_condition=boundary_condition,
                               n_point_stencil=n_point_stencil)

    # fill d_list and E_list
    for d in np.linspace(0, 10, 50):
        d_list.append(d)
        potential_fn = functools.partial(two_poschl_teller, d=d)
        solver.update_potential(potential_fn)
        solver.solve_ground_state()
        energy = solver.eigenvalues[0]
        E_list.append(energy)

    # save E vs. d into d_E_table.dat
    import csv

    with open('d_E_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d', 'E'])
        writer.writerows(zip(d_list, E_list))

    # plot E vs d
    plot_and_save(d_list, E_list, 'd', 'E', 'E vs d')

    # plot potential, wave function and eigenvalue at d = test_d
    test_d = 2
    potential_fn = functools.partial(two_poschl_teller, d=test_d)
    solver.update_potential(potential_fn)
    solver.solve_ground_state()
    plot_list = [['potential', grids, potential_fn(grids), '-', 'orange'],
                 ['wave function', grids, solver.wave_function[0], '-', 'blue'],
                 ['eigenvalue', grids,
                  np.full(len(grids), solver.eigenvalues[0]), '--', 'green']]
    plot_multiple_and_save(plot_list, f'd={test_d}', ylim=(-2, 1))
