import single_electron, ext_potentials

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


# plotting parameters
params = {'mathtext.default': 'default'}
plt.rcParams.update(params)
plt.rcParams['axes.axisbelow'] = True
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
fig, ax = plt.subplots()

# suggested values for exponential hydrogenic systems,
# see Tom Baker et. al. PHYSICAL REVIEW B 91, 235141 (2015)
pi = np.pi
A = 1.071295
k_inv = 2.385345
k = 1. / k_inv


# continuous exponential well ------------------------------
# see RP_logbook 7/29/19

num_electrons = 4

root_list = ext_potentials.exp_hydro_open_roots(A,k)
energy_list = [-(1 / 8) * k ** 2 * x ** 2 for x in root_list]
E_inf_th = energy_list[0]
print('E_inf_th = ', E_inf_th-E_inf_th) # theoretical/analytical value

d = 7
root_list = ext_potentials.exp_hydro_cont_well_roots(A, k, d=d)
energy_list = [-(1 / 8) * k ** 2 * x ** 2 for x in root_list]
E_cont_th = energy_list[0]
print('E_exp_th = ', E_cont_th-E_inf_th) # theoretical/analytical value


grid = 51
grids = np.linspace(-d / 2, d / 2, grid)
solver = single_electron.EigenSolver(grids, potential_fn=functools.partial(
    ext_potentials.exp_hydro_cont_well, A=A, k=k, d=d, a=0), boundary_condition='closed', n_point_stencil=3)
solver.solve_ground_state()
E_closed = solver.eigenvalues[0]
print('E_closed = ', E_closed-E_inf_th)
psi_closed = plt.plot(grids,-solver.wave_function[0], label = 'closed boundary')


grid = 101
grids = np.linspace(-d / 2, d / 2, grid)
solver = single_electron.EigenSolver(grids, potential_fn=functools.partial(
    ext_potentials.exp_hydro_cont_well, A=A, k=k, d=d, a=0), boundary_condition='exponential decay', n_point_stencil=3, approx_E=E_closed)
solver.solve_ground_state()
E_exp = solver.eigenvalues[0]
print('E_exp = ', E_exp-E_inf_th)
psi_exp = plt.plot(solver.extended_grids,solver.extended_wave_function, label='exp-decay boundary')


grid = 101
grids = np.linspace(-10, 10, grid)
solver = single_electron.EigenSolver(grids, potential_fn=functools.partial(
    ext_potentials.exp_hydrogenic, A=A, k=k, a=0), boundary_condition='open', n_point_stencil=3, approx_E=E_exp)
solver.solve_ground_state()
psi_open = plt.plot(grids,solver.wave_function[0], label='infinite system')


ax.set_xlabel("$x$", fontsize=18)
ax.set_ylabel("$\psi(x)$", fontsize=18)


plt.legend(fontsize=16, loc='upper right')
plt.show()
