import single_electron, functionals, ext_potentials
import blue_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys


def get_n_r0(grids, n):
    n_r0 = []
    for i, r0 in enumerate(grids):
        solver = single_electron.EigenSolver(grids,
                                             potential_fn=functools.partial(
                                                 blue_potentials.blue_helium_1d_erf,
                                                 r0=r0, n_r=n[i]))

        solver.solve_ground_state()
        n_r0.append(solver.density)

    n_r0 = np.asarray(n_r0)
    return n_r0


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h

    n_dmrg = np.load('densities.npy')[0]

    n_r0 = get_n_r0(grids, n_dmrg)

    np.save("n_r0_1D_He_erf_gam_1or_s.npy", n_r0)
