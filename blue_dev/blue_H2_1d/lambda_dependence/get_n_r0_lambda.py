import single_electron, ext_potentials, functionals
import blue_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys
import multiprocessing as mp
import os

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
'''


def get_v_ext_lambda(grids, blue_potential, lam):
    # see logbook 4/8/20: 'adabatic connection blue He revisited
    return blue_potential + 0.5 * (1 - lam) * functionals.hartree_potential(
        grids, n)


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h
    potentials = np.load("H2_data/potentials.npy")
    potentials = potentials[:40]

    n_r0_R = []
    for i, potential in enumerate(potentials):
        print(i)
        n_r0 = []
        for r0 in grids:
            solver = single_electron.EigenSolver(grids,
                                                 potential_fn=functools.partial(
                                                     get_blue_potential,
                                                     pot=potential, r0=r0),
                                                 boundary_condition='open',
                                                 num_electrons=1)
            solver.solve_ground_state()

            E0 = solver.eigenvalues[0]
            # print("E0 = ", E0)

            n = solver.density

            n_r0.append(n)

        n_r0 = np.asarray(n_r0)
        n_r0_R.append(n_r0)

    n_r0_R = np.asarray(n_r0_R)
    np.save('n_r0_R', n_r0_R)
