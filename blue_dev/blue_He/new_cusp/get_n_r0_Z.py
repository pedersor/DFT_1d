import single_electron, blue_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys


def get_grids(Z):
    # different grids are used for different Z = 1,2,...,6
    if Z == 1:
        L = 10
        grid_size = 1000
        grids = np.linspace(L / grid_size - 0.0004, L, grid_size)
        return grids
    elif Z == 2:
        L = 20
        # for 1000, L = 20 using initial pt 0.018435 to minimize error of H and He
        grid_size = 1000
        grids = np.linspace(0.018435, L, grid_size)
        return grids
    else:
        if Z not in [1, 2, 3, 4, 6]:
            print('Z should be in [1, 2, 3, 4, 6]')
            return
        else:
            # density concentrated near nucleus, so smaller grid size
            L = 5
            grid_size = 1000
            grids = np.linspace(L / grid_size - 0.0004, L, grid_size)
            return grids


if __name__ == '__main__':

    Z_list = np.linspace(2, 8, 7)
    Z_list = [2]

    # gets cusp condition right (previously off by factor of 2)
    new_cusp_cond = 0.5

    n_r0_Z = []
    for Z in Z_list:
        grids = get_grids(Z)
        h = grids[1] - grids[0]

        n_r0 = []
        print("Z = ", str(Z))
        for i, r0 in enumerate(grids):
            print('i, r0 = ', i, '  ', r0)
            solver = single_electron.EigenSolver(grids,
                                                 potential_fn=functools.partial(
                                                     blue_potentials.blue_helium_spherical_erf,
                                                     gam=3., r0=r0, Z=Z),
                                                 boundary_condition='open',
                                                 num_electrons=1)
            solver.solve_ground_state()

            # E0 = solver.eigenvalues[0]
            # print("E0 = ", E0)

            wf0 = solver.wave_function[0]
            wf0_unnorm = wf0 / grids

            coeff_norm = np.sqrt((1. / (4 * (np.pi) * np.sum(
                grids * grids * wf0_unnorm * wf0_unnorm) * h)))
            wf0 = coeff_norm * wf0_unnorm

            n = wf0 * wf0
            # print("n integral check: ", 4 * (np.pi) * np.sum(grids * grids * n) * h)
            # plt.plot(grids, np.abs(wf0), label='$r_0$ = ' + str(r0))

            n_r0.append(n)

        n_r0 = np.asarray(n_r0)
        n_r0_Z.append(n_r0)

    n_r0_Z = np.asarray(n_r0_Z)
    print(n_r0_Z)
    np.save("n_r0_Z_blue_gam_3.npy", n_r0_Z)

sys.exit()

if __name__ == '__main__':

    Z_list = np.linspace(2, 8, 7)
    Z_list = [2]

    # gets cusp condition right (previously off by factor of 2)
    new_cusp_cond = 0.5

    n_r0_Z = []
    for Z in Z_list:
        grids = get_grids(Z)
        h = grids[1] - grids[0]

        n_r0 = []
        print("Z = ", str(Z))
        for r0 in grids:
            solver = single_electron.EigenSolver(grids,
                                                 potential_fn=functools.partial(
                                                     blue_potentials.blue_helium,
                                                     r0=r0, Z=Z,
                                                     lam=new_cusp_cond),
                                                 boundary_condition='open',
                                                 num_electrons=1)
            solver.solve_ground_state()

            # E0 = solver.eigenvalues[0]
            # print("E0 = ", E0)

            wf0 = solver.wave_function[0]
            wf0_unnorm = wf0 / grids

            coeff_norm = np.sqrt((1. / (4 * (np.pi) * np.sum(
                grids * grids * wf0_unnorm * wf0_unnorm) * h)))
            wf0 = coeff_norm * wf0_unnorm

            n = wf0 * wf0
            # print("n integral check: ", 4 * (np.pi) * np.sum(grids * grids * n) * h)
            # plt.plot(grids, np.abs(wf0), label='$r_0$ = ' + str(r0))

            n_r0.append(n)

        n_r0 = np.asarray(n_r0)
        n_r0_Z.append(n_r0)

    n_r0_Z = np.asarray(n_r0_Z)
    print(n_r0_Z)
    np.save("n_r0_Z_blue_gam_0.npy", n_r0_Z)
