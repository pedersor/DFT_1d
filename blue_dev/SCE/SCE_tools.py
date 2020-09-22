import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import InterpolatedUnivariateSpline
import single_electron, ext_potentials
import functools
from scipy import special

def exp_hydrogenic(grids, A=1.071295, k=(1. / 2.385345), Z=1):
    vp = -Z * A * np.exp(-k * np.abs(grids))
    return vp


def cumulant_funct_1d(grids, n):
    # cumulant function for 1D density
    N_e = []
    for i in range(len(grids)):
        N_e.append(np.trapz(n[:i], grids[:i]))

    N_e = np.asarray(N_e)

    return N_e


def get_inv_N_e(N_e, value):
    # returns index (idx) of closest grid value that yields N_e.
    # e.g. inv_N_e(2.0) = grid[idx] -> N_e(grid[idx]) = 2.0.

    idx = np.searchsorted(N_e, value, side="left")
    if idx > 0 and (
            idx == len(N_e) or math.fabs(value - N_e[idx - 1]) < math.fabs(
        value - N_e[idx])):
        return idx - 1
    else:
        return idx


def get_f_i_1d(grids, i, n, N):
    N_e = cumulant_funct_1d(grids, n)

    a_bar_i = get_inv_N_e(N_e, N - i)
    f_i = []
    for j, grid in enumerate(grids):
        if j <= a_bar_i:
            f_i.append(grids[get_inv_N_e(N_e, N_e[j] + i)])
        else:
            f_i.append(grids[get_inv_N_e(N_e, N_e[j] + i - N)])

    f_i = np.asarray(f_i)
    return f_i


def example_density_1d(grids, lam=0.5):
    # example density N = 5 from https://journals.aps.org/pra/pdf/10.1103/PhysRevA.60.4387
    return 5 * lam * (np.pi) ** (-0.5) * np.exp(-1 * (lam * grids) ** 2)


def get_v_Hxc_SCE_1d(grids, f_i):
    A = 1.071295
    k = (1. / 2.385345)

    integrand = -1 * (k * (grids - f_i) / np.abs(
        grids - f_i)) * ext_potentials.exp_hydrogenic(grids - f_i)

    return cumulant_funct_1d(grids, integrand)

def get_hookes_atom_density(r):
    # 3D hooke's atom radial density

    n = (2 / ((np.pi ** (3 / 2)) * (8 + 5 * (np.pi ** 0.5)))) * np.exp(
        -0.5 * (r ** 2)) * (((np.pi / 2) ** (0.5)) * (
            (7 / 4) + (0.25 * (r ** 2)) + (r + (1 / r)) * special.erf(
        r / np.sqrt(2))) + np.exp(-0.5 * (r ** 2)))

    n[0] = 0.0893193  # take the limit for r -> 0
    return n

def get_f_i_radial(grids, i, n, N):
    N_e = cumulant_funct_1d(grids, n*grids*grids*4*np.pi)

    plt.plot(grids, N_e)
    plt.show()

if __name__ == '__main__':
    # '1d', 'hookes_atom'
    run = 'hookes_atom'

if run == 'hookes_atom':
    grids = np.arange(0, 5000) * 0.005
    n = get_hookes_atom_density(grids)

    i = 1
    N = 2

    get_f_i_radial(grids, i, n, N)


if run == '1d':
    # 1d He example
    n = np.load('He_solo/densities.npy')[0]
    grids = np.arange(-256, 257) * 0.08

    n_interp = InterpolatedUnivariateSpline(grids, n, k=3)
    grids_interp = np.arange(-2560, 2561) * 0.008
    n = n_interp(grids_interp)
    grids = grids_interp

    '''
    # example density from paper
    grids = np.arange(-256, 257) * (0.08/4)
    n = example_density(grids)
    '''

    N = 2
    lam = 1

    f_i = get_f_i_1d(grids, 1, n, N)

    v_Hxc_SCE = get_v_Hxc_SCE_1d(grids, f_i)

    f_i_blue = []

    for r_idx, grid in enumerate(grids):
        v_SCE_blue = np.zeros(len(grids))
        # plt.scatter(grids, f_i, label='$f_' + str(i) + '$', marker='.')

        v_SCE_blue += lam * v_Hxc_SCE - lam * exp_hydrogenic(
            grids - grid)

        v_SCE_blue_min_idx = np.argmin(v_SCE_blue)

        f_i_blue.append(grids[v_SCE_blue_min_idx])

        # print('f_' + str(i) + ' = ' + str(f_i[r_idx]))
        # print('index: ', np.searchsorted(grids, f_i[r_idx], side="left"))

    f_i_blue = np.asarray(f_i_blue)

    plt.scatter(grids, f_i_blue, label='blue', marker='.')
    plt.scatter(grids, f_i, label='exact $f_1$', marker='.')

    plt.xlabel('$x$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()
