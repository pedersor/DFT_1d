import sys

# linux cluster path
sys.path.append('/DFS-B/DATA/burke/pedersor/Kohn_Sham_DFT_1d')
sys.path.append('/DFS-B/DATA/burke/pedersor/Kohn_Sham_DFT_1d/blue_dev')

import single_electron, functionals
import hf_scf
import blue_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import multiprocessing as mp
import os


def get_v_ext(grids, potential):
    return potential


def get_n_HF(grids, potential, DMRG_density=None):
    v_ext = functools.partial(get_v_ext, potential=potential)
    v_h = functools.partial(functionals.hartree_potential)
    fock_op = functionals.fock_operator(grids=grids)

    solver = hf_scf.HF_Solver(grids, v_ext=v_ext, v_h=v_h,
                              fock_operator=fock_op, num_electrons=2)
    solver.solve_self_consistent_density(sym=1)

    # Non-Interacting Kinetic Energy
    print("T_s =", solver.T_s, flush=True)

    # External Potential Energy
    print("V =", solver.V, flush=True)

    # Hartree Energy
    print("U =", solver.U, flush=True)

    # Exchange Energy
    print("E_x =", solver.E_x, flush=True)

    # Total Energy
    print("E =", solver.E_tot, flush=True)

    return solver.density


def get_v_ext_lambda(grids, blue_potential, n, lam):
    # see logbook 4/8/20: 'adabatic connection blue He revisited
    return blue_potential + 0.5 * (1 - lam) * functionals.hartree_potential(
        grids, n)


def get_n_r0_lambda(lam, grids, pot, n):
    n_r0 = []
    print("lambda = ", str(lam), flush=True)
    for r0 in grids:
        blue_potential = blue_potentials.pure_blue_arb_pot_1d(grids, pot, r0, lam)
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


# bash sed problem with indented variables TODO: cleanup
pot_idx = 51
if __name__ == '__main__':
    try:
        slurm_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    except:
        slurm_cpus = 1

    print("Using %s cores" % slurm_cpus, flush=True)
    pool = mp.Pool(slurm_cpus)

    h = 0.08
    grids = np.arange(-256, 257) * h

    # H2_data/ not on github due to data limits
    pot = np.load("H2_data/potentials.npy")[pot_idx - 1]

    n_HF = get_n_HF(grids, pot)

    lambda_list = np.linspace(0, 1, 11)

    n_r0 = functools.partial(get_n_r0_lambda, grids=grids, pot=pot, n=n_HF)

    n_r0_lambda = pool.map(n_r0, lambda_list)

    n_r0_lambda = np.asarray(n_r0_lambda)
    np.save("n_r0_lambda_HF.npy", n_r0_lambda)
