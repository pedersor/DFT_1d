import numpy as np
from ext_potentials import poschl_teller
import single_electron
import matplotlib.pyplot as plt
import functools
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


def plot_and_save(x_list, y_list, x_name, y_name, folder, name):
    
    # initialize figure for plots
    fig, ax = get_plotting_params()
    # matplotlib trick to obtain same color of a previous plot
    ax.plot(x_list, y_list, marker='o', linestyle='solid', color='blue')
        
    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)

    plt.grid(alpha=0.4)
    plt.gca().xaxis.grid(True, which='minor', alpha=0.4)
    plt.gca().yaxis.grid(True, which='minor', alpha=0.4)
    
    # create folder if no such directory
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    # save fig
    plt.savefig(f'{folder}/{name}.png')
    plt.close()


def generate_potential(grids, d, lam=1., a=1.):
    
    potential_grids = poschl_teller(grids, lam = lam, a = a, center = -d/2)
    potential_grids += poschl_teller(grids, lam = lam, a = a, center = d/2)
    
    return potential_grids

if __name__ == '__main__':

    test_range = (-20, 20)
    grids = np.linspace(*test_range, 1000)
    boundary_condition = 'closed'
    n_point_stencil = 5
    
    d_list = []
    E_list = []
    
    for d in range(0, 40):
        
        d = d/2
        
        d_list.append(d)
        
        potential_fn = functools.partial(generate_potential, d = d)
        solver = single_electron.SparseEigenSolver(grids,
                                                  potential_fn = potential_fn,
                                                  boundary_condition = boundary_condition,
                                                  n_point_stencil = n_point_stencil)
        
        solver.solve_ground_state()
        energy = solver.eigenvalues[0]
        E_list.append(energy)
        
        #potential_grids = potential_fn(grids)
        
    plot_and_save(d_list, E_list, 'd', 'E', 'two_poschl_teller', 'E vs d')
        
        
        
        