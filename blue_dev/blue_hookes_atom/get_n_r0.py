import single_electron, blue_potentials, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
from scipy import special
import sys


def get_n(r):
    # 3D hooke's atom radial density

    n = (2 / ((np.pi ** (3 / 2)) * (8 + 5 * (np.pi ** 0.5)))) * np.exp(
        -0.5 * (r ** 2)) * (((np.pi / 2) ** (0.5)) * (
            (7 / 4) + (0.25 * (r ** 2)) + (r + (1 / r)) * special.erf(
        r / np.sqrt(2))) + np.exp(-0.5 * (r ** 2)))

    n[0] = 0.0893193  # take the limit for r -> 0
    return n


if __name__ == '__main__':
    # sphericalized exact

    grids = np.linspace(0.0056, 6, 1000)
    h = grids[1] - grids[0]

    exact_density = get_n(grids)

    n_r0 = []
    for i, r0 in enumerate(grids):
        # exact sphericalized v_s
        # sph_v_s = blue_potentials.exact_sph_hookes_atom(grids, r2=r0)

        gam = (((4 * np.pi * exact_density[i]) / 3) ** (1 / 3))

        pot = blue_potentials.blue_helium_spherical_erf(grids, r0=r0, gam=gam,
                                                        Z=0) + ext_potentials.harmonic_oscillator(
            grids, k=1 / 4)

        solver = single_electron.EigenSolver(grids,
                                             potential_fn=functools.partial(
                                                 ext_potentials.get_gridded_potential,
                                                 potential=pot),
                                             boundary_condition='open',
                                             num_electrons=1)
        solver.solve_ground_state()

        E0 = solver.eigenvalues[0]
        print("E0 = ", E0)

        wf0 = solver.wave_function[0]
        wf0_unnorm = wf0 / grids

        coeff_norm = np.sqrt((1. / (4 * (np.pi) * np.sum(
            grids * grids * wf0_unnorm * wf0_unnorm) * h)))
        wf0 = coeff_norm * wf0_unnorm

        n = wf0 * wf0
        # print("n integral check: ", 4 * (np.pi) * np.sum(grids * grids * n) * h)
        # plt.plot(grids, n, label='$r_0$ = ' + str(r0))
        # plt.show()

        n_r0.append(n)

    n_r0 = np.asarray(n_r0)

    np.save("n_r0_blue_gam_1or_s.npy", n_r0)

sys.exit()
