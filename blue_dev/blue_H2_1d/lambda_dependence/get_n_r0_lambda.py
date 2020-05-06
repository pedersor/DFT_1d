import single_electron, functionals
import blue_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys
import multiprocessing as mp
import os


def get_v_ext_lambda(grids, blue_potential, n, lam):
    # see logbook 4/8/20: 'adabatic connection blue He revisited
    return blue_potential + 0.5 * (1 - lam) * functionals.hartree_potential(
        grids, n)


def get_n_r0_lambda(lam, grids, pot, n):
    n_r0 = []
    print("lambda = ", str(lam), flush=True)
    for r0 in grids:
        new_factor = 0.5
        blue_potential = blue_potentials.blue_H2_1d(grids, pot, r0,
                                                    lam=new_factor)
        solver = single_electron.EigenSolver(grids,
                                             potential_fn=functools.partial(
                                                 get_v_ext_lambda,
                                                 blue_potential=blue_potential,
                                                 n=n, lam=lam))

        solver.solve_ground_state()
        n_r0.append(solver.density)

    n_r0 = np.asarray(n_r0)
    print('done ', str(lam), flush=True)
    return n_r0


if __name__ == '__main__':
    slurm_cpus = 1  # int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    print("Using %s cores" % slurm_cpus, flush=True)
    pool = mp.Pool(slurm_cpus)

    h = 0.08
    grids = np.arange(-256, 257) * h
    potentials = np.load("../H2_data/potentials.npy")
    densities = np.load("../H2_data/densities.npy")

    # get H2 values
    pot = potentials[0]
    n = densities[0]

    lambda_list = np.linspace(0, 1, 11)
    lambda_list = np.array([1])

    n_r0 = functools.partial(get_n_r0_lambda, grids=grids, pot=pot, n=n)

    n_r0_lambda = pool.map(n_r0, lambda_list)

    n_r0_lambda = np.asarray(n_r0_lambda)
    np.save("n_r0_1D_He.npy", n_r0_lambda)
