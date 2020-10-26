import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import InterpolatedUnivariateSpline
import single_electron, ext_potentials, functionals
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


def get_derivative_mat(N, h):
    mat = np.eye(N)
    idx = np.arange(N)

    A = [0, 2 / 3, -1 / 12]

    for j, A_n in enumerate(A):
        mat[idx[j:], idx[j:] - j] = -A_n
        mat[idx[:-j], idx[:-j] + j] = A_n

    return mat / h


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


def get_v_Hxc_SCE_1d(grids, f_1):
    A = 1.071295
    k = (1. / 2.385345)

    integrand = -1 * (k * (grids - f_1) / np.abs(
        grids - f_1)) * ext_potentials.exp_hydrogenic(grids - f_1)

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
    N_e = cumulant_funct_1d(grids, n * grids * grids * 4 * np.pi)

    plt.plot(grids, N_e)
    plt.show()


def get_split_discont_index(data, stepsize=2):
    return np.where(np.abs(np.diff(data)) > stepsize)

def harmonic_osc_pot(grids, k=0.25):
    return 0.5 * k * (grids ** 2)


if __name__ == '__main__':
    # '1d', 'hookes_atom'
    run = '1d'

if run == 'hookes_atom':
    grids = np.arange(0, 5000) * 0.005
    n = get_hookes_atom_density(grids)

    i = 1
    N = 2

    get_f_i_radial(grids, i, n, N)

if run == '1d':
    # 1d He example
    n = np.load('He_solo/densities.npy')[0]
    #n = np.load('H2_data/densities.npy')[74]
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
    lam = 50

    f_i = get_f_i_1d(grids, 1, n, N)

    v_Hxc_SCE_exact = get_v_Hxc_SCE_1d(grids, f_i)  # exact

    '''
    v_H_SCE = functionals.hartree_potential(grids, n)  # Hartree piece
    v_Hxc_SCE = -0.5 * v_H_SCE  # initial guess
    #np.save('v_H_SCE.npy', v_H_SCE)
    '''

    v_Hxc_SCE = -0.5 * np.load('v_H_SCE.npy')

    num_iter = 1

    for iter in range(num_iter):
        if iter != 0:
            f_i_prev = f_i_blue
        else:
            f_i_prev = 0

        f_i_blue = []
        for r_idx, grid in enumerate(grids):
            v_SCE_blue = np.zeros(len(grids))

            v_SCE_blue += v_Hxc_SCE - lam * exp_hydrogenic(
                grids - grid)

            v_SCE_blue_min_idx = np.argmin(v_SCE_blue)

            f_i_blue.append(grids[v_SCE_blue_min_idx])

        f_i_blue = np.asarray(f_i_blue)

        split_idx = 2560  # get_split_discont_index(f_i_blue)
        print(split_idx)

        print(get_split_discont_index(f_i_blue))

        n_over_nf = cumulant_funct_1d(grids, n / (n_interp(f_i_blue) + 0.00001))
        n_over_nf[split_idx:] = n_over_nf[split_idx:] - n_over_nf[-1]

        residual = n_over_nf - f_i_blue

        f_i_new = 0.5 * f_i_blue + 0.5 * (f_i_blue + residual)

        v_Hxc_SCE = get_v_Hxc_SCE_1d(grids, f_i_new)

        if iter == 0:
            plt.scatter(grids, f_i_blue, label='k = ' + str(iter), marker='.')


    #plt.scatter(grids, f_i_blue, label='k = ' + str(iter), marker='.')
    plt.scatter(grids, f_i, label='exact f', marker='.')
    plt.xlabel('$x$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()
