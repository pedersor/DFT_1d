import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import math
import ext_potentials


def get_cumulant_funct_1d(grids, n):
    # cumulant function for 1D density:
    # N_e(x) = \int_{-\infty}^x dx' n(x')

    # this could be maybe more accurate ...
    # N_e2 = np.asarray([n.integral(grids[0], grid) for grid in grids])

    N_e = np.asarray([np.trapz(n[:j], grids[:j]) for j, _ in enumerate(grids)])

    return N_e


def get_inv_cumulant_funct_1d(N_e, value):
    # inverse cumulant function for 1D density:
    # returns index (idx) of closest grid value that yields N_e.
    # e.g. inv_N_e(2.0) = grid[idx] -> N_e(grid[idx]) = 2.0.

    idx_left = np.searchsorted(N_e - 0.0000000000001, value, side="left")
    if idx_left > 0 and (
            idx_left == len(N_e) or math.fabs(
        value - N_e[idx_left - 1]) < math.fabs(
        value - N_e[idx_left])):
        idx_left = idx_left - 1

    idx_right = np.searchsorted(N_e + 0.0000000000001, value, side="right")
    if idx_right > 0 and (
            idx_right == len(N_e) or math.fabs(
        value - N_e[idx_right - 1]) < math.fabs(
        value - N_e[idx_right])):
        idx_right = idx_right - 1

    idx = idx_left + int((idx_right - idx_left) / 2)
    return idx


def get_f_i_1d(grids, i, n, N):
    # see https://research.vu.nl/ws/portalfiles/portal/95269314/67785.pdf
    # Eqs. 3.31 - 3.32. Below follows same notation as these equations.

    N_e = get_cumulant_funct_1d(grids, n)

    a_bar_i = get_inv_cumulant_funct_1d(N_e, N - i)

    f_i = []
    for j, grid in enumerate(grids):
        if j <= a_bar_i:
            f_i.append(grids[get_inv_cumulant_funct_1d(N_e, N_e[j] + i)])
        else:
            f_i.append(grids[get_inv_cumulant_funct_1d(N_e, N_e[j] + i - N)])

    f_i = np.asarray(f_i)
    return f_i


def get_v_ee_SCE_1d(grids, n, f_i_lst):
    integrand = np.zeros(len(grids))
    for f_i in f_i_lst:
        integrand += -1 * ext_potentials.exp_hydrogenic(grids - f_i)
    integrand *= n
    v_ee_SCE = 0.5 * np.trapz(integrand, grids)
    return v_ee_SCE


def get_Vee_SCE(grids, n, N=2):
    f_i_exact = []

    for i in np.arange(1, N):
        f_i = get_f_i_1d(grids, i, n, N)
        f_i_exact.append(f_i)

    f_i_exact = np.asarray(f_i_exact)

    v_ee_SCE_exact = get_v_ee_SCE_1d(grids, n, f_i_exact)
    return f_i_exact, v_ee_SCE_exact


if __name__ == '__main__':
    # 1d He example

    N = 2  # num of electrons
    # exact density
    n = np.load('He_solo/densities.npy')[0]
    # corresponding 1d grid used
    grids = np.arange(-256, 257) * 0.08

    # simple interpolation gives finer resolution of co-motion ftns.
    n_interp = InterpolatedUnivariateSpline(grids, n, k=3)
    # evaluate on denser grid
    grids_interp = np.arange(-2560, 2561) * 0.008
    n = n_interp(grids_interp)
    grids = grids_interp

    for i in range(N):
        f_i = get_f_i_1d(grids, i, n, N)
        # use scatter since divergence at origin is messy..
        plt.scatter(grids, f_i, marker='.', label='$f_' + str(i) + '$')

    plt.xlabel('$x$', fontsize=16)
    plt.xlim(-5, 5)
    plt.grid(alpha=0.4)
    plt.legend(fontsize=16)
    plt.show()
