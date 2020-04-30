import single_electron, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys

# plotting parameters
params = {'mathtext.default': 'default'}
plt.rcParams.update(params)
plt.rcParams['axes.axisbelow'] = True
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
fig, ax = plt.subplots()


def get_blue_potential(grids, pot, r0):
    return pot - ext_potentials.exp_hydrogenic(grids - r0)


def get_He_pot(grids):
    return ext_potentials.exp_hydrogenic(grids, Z=2)


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h
    #potentials = np.load("H2_data/potentials.npy")
    potentials = [get_He_pot(grids)]

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
    np.save('n_r0_0', n_r0_R)
    #np.save('n_r0_R', n_r0_R)
