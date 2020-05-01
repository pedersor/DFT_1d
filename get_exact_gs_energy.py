import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import six
from six.moves import range
import copy
import timeit
import functools

import single_electron
import ext_potentials

def get_ground_state(potential_fn, range, num_grid = 5000, n_point_stencil=5):
    # approximate exact potential by using grids of default 5000
    
    grids = np.linspace(*range, num_grid)
    solver = single_electron.EigenSolver(grids, 
                                         potential_fn= potential_fn, 
                                         boundary_condition='closed',
                                         n_point_stencil=5)
    
    solver.solve_ground_state()

    # return lowest eigenvalue from FDM
    return np.array([solver.eigenvalues[0], solver.wave_function[0], grids])

def write_ground_state(potential_fn, func_name, range, num_grid = 5000, n_point_stencil = 5):
    
    energy_wf_array = get_ground_state(potential_fn, *range, num_grid, n_point_stencil)
    
    np.save(f'ground_states_npy/{func_name}', energy_wf_array)
    
    
def get_exact_energy_from_npy(potential_fn, range, num_grid = 5000, n_point_stencil = 5, rewrite = False, file_name = None):
    
    if file_name:
        func_name = file_name
    else:
        try:
            func_name = potential_fn.__name__
        except AttributeError:
            func_name = potential_fn.func.__name__
    
    if rewrite:
        write_ground_state(potential_fn, func_name, range, num_grid, n_point_stencil)
        return np.real(np.load(f'ground_states_npy/{func_name}.npy', allow_pickle=True)[0])
    
    else:
        try:
            return np.real(np.load(f'ground_states_npy/{func_name}.npy', allow_pickle=True)[0])
        
        except FileNotFoundError:
            write_ground_state(potential_fn, func_name, range, num_grid, n_point_stencil)
            return np.real(np.load(f'ground_states_npy/{func_name}.npy', allow_pickle=True)[0])

def get_gs_wf_from_npy(potential_fn, range, num_grid = 5000, n_point_stencil = 5, rewrite = False, file_name = None):
    
    if file_name:
        func_name = file_name
    else:
        try:
            func_name = potential_fn.__name__
        except AttributeError:
            func_name = potential_fn.func.__name__
    
    if rewrite:
        write_ground_state(potential_fn, func_name, range, num_grid, n_point_stencil)
        return np.real(np.load(f'ground_states_npy/{func_name}.npy', allow_pickle=True)[1])
    
    else:
        try:
            return np.real(np.load(f'ground_states_npy/{func_name}.npy', allow_pickle=True)[1])
        
        except FileNotFoundError:
            write_ground_state(potential_fn, func_name, range, num_grid, n_point_stencil)
            return np.real(np.load(f'ground_states_npy/{func_name}.npy', allow_pickle=True)[1])