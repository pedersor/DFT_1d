import ext_potentials
import numpy as np
from scipy import special
from scipy import integrate
from scipy import optimize
import math
import matplotlib.pyplot as plt
import sys


def sph_blue_impurity(grids, r0, lam=1.):
    vp = []
    for r in grids:
        if np.abs(r) < r0:
            vp.append((lam / r0))
        else:
            vp.append((lam) / np.abs(r))

    return np.asarray(vp)


def blue_helium(grids, r0, Z=2, lam=1.):
    vp = []
    for r in grids:
        if np.abs(r) < r0:
            vp.append((-Z / np.abs(r)) + (lam / r0))
        else:
            vp.append((-Z + lam) / np.abs(r))

    return np.asarray(vp)


def blue_helium_spherical_erf(grids, r0, gam=1., Z=2, lam=1.):
    vp = []

    term_1 = np.exp(-1 * (gam * (grids + r0)) ** 2)
    term_2 = -1 * np.exp(
        (-1 * (gam * (grids + r0)) ** 2) + (4 * grids * r0 * (gam ** 2)))

    term_3 = (gam * (np.pi ** 0.5)) * (
            (-grids + r0) * special.erf(gam * (grids - r0)) + (
            grids + r0) * special.erf(gam * (grids + r0)))

    spherical_erf = term_1 + term_2 + term_3

    spherical_erf = spherical_erf / (gam * grids * r0 * (np.pi ** 0.5) * 2)

    for i, r in enumerate(grids):
        if np.abs(r) < r0:
            vp.append(
                (-Z / np.abs(r)) + 0.5 * (lam / r0) + 0.5 * lam * spherical_erf[
                    i])
        else:
            vp.append(
                ((-Z + 0.5 * lam) / np.abs(r)) + 0.5 * lam * spherical_erf[i])

    return np.asarray(vp)


def blue_helium_1d_erf(grids, r0, n_r, Z=2):
    # 1/(2|r-r'|) for r ~ r' and 1/|r-r'| for large separations of r and r'.
    # Decrease/increase 'transition time' using gamma.

    # \gamma = 1.5/r_s
    gam = (3 / 2) * (((4 * np.pi * n_r) / 3) ** (1 / 3))

    return ext_potentials.exp_hydrogenic(grids,
                                         Z=Z) - 0.5 * ext_potentials.exp_hydrogenic(
        grids - r0) * (1 + special.erf(np.abs(gam * (grids - r0))))


def blue_1d_H2_erf(grids, r0, n_r, pot):
    # 1/(2|r-r'|) for r ~ r' and 1/|r-r'| for large separations of r and r'.
    # Decrease/increase 'transition time' using gamma.

    # \gamma = 1.5/r_s
    gam = (3 / 2) * (((4 * np.pi * n_r) / 3) ** (1 / 3))

    ext_pot = ext_potentials.get_gridded_potential(grids, pot)

    return ext_pot - 0.5 * ext_potentials.exp_hydrogenic(
        grids - r0) * (1 + special.erf(np.abs(gam * (grids - r0))))


def blue_helium_1d(grids, r0, Z=2, lam=1):
    return ext_potentials.exp_hydrogenic(grids,
                                         Z=Z) - lam * ext_potentials.exp_hydrogenic(
        grids - r0)


def helium(grids, Z=2):
    return -Z / (np.abs(grids))


def blue_H2_1d(grids, pot, r0, lam=1):
    # pot is the pre-generated external potential of the separated H2
    return pot - lam * ext_potentials.exp_hydrogenic(grids - r0)


def exact_sph_hookes_atom(grids, r2):
    # slightly offset so no numerical errors
    r2 += 0.000001

    def u(r1, r2, theta):
        return np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(theta))

    def get_v_s(r1, r2, theta):
        numerator_1 = 0.125 * (r1 ** 6) + r2 ** 2 - 0.75 * (r2 ** 4)
        num_2 = (r1 ** 4) * (-1.25 + 0.5 * (r2 ** 2)) + (r1 ** 2) * (
                1 - 4 * (r2 ** 2) + 0.125 * (r2 ** 4))
        num_3 = (0.25 * (r1 ** 4) - 1.5 * (r2 ** 2) + (r1 ** 2) * (
                -1.5 + 0.25 * (r2 ** 2))) * u(r1, r2, theta)
        num_4 = r1 * r2 * np.cos(theta) * (
                -2 - 0.5 * (r1 ** 4) + 3.5 * (r2 ** 2) + 3 * u(r1, r2, theta))
        num_5 = r1 * r2 * np.cos(theta) * (
                (r1 ** 2) * (4.5 - 0.5 * (r2 ** 2) - 0.5 * u(r1, r2, theta)))
        num_6 = (r1 ** 2) * (-2 + 0.25 * (r1 ** 2)) * (r2 ** 2) * np.cos(
            2 * theta)

        numerator = numerator_1 + num_2 + num_3 + num_4 + num_5 + num_6

        denominator = u(r1, r2, theta) * (u(r1, r2, theta) ** 2) * (
                2 + u(r1, r2, theta))

        return numerator / denominator

    theta = np.linspace(0, np.pi, 100)
    sph_v_s = []
    for r1 in grids:
        v_s = get_v_s(r1, r2, theta)
        sph_v_s.append(np.trapz(v_s * np.sin(theta), theta))
    sph_v_s = np.asarray(sph_v_s)

    return 0.5 * sph_v_s


if __name__ == '__main__':
    # 1d He erf calc

    h = 0.08
    grids = np.arange(-256, 257) * h

    densities = np.load('blue_He_1d/densities.npy')[0]

    plt.plot(grids, densities)
    plt.show()

    erf_1d = blue_helium_1d_erf(grids, r0=grids[280], n_r=densities[280])

    plt.plot(grids, erf_1d)
    plt.show()

    sys.exit()
    # hooke's atom testing

    grids = np.linspace(0.01, 5, 1000)
    sph_v_s = exact_sph_hookes_atom(grids, r2=grids[100])
    plt.plot(grids, sph_v_s)

    pot = sph_blue_impurity(grids,
                            r0=grids[100],
                            lam=1 / 2) + ext_potentials.harmonic_oscillator(
        grids, k=1 / 4)

    plt.plot(grids, pot - 1)

    # plt.plot(grids, blue_helium(grids, r0=0.5, Z=0, lam=0.5) - 1)

    erf_plot = blue_helium_spherical_erf(grids, r0=0.5, Z=0, gam=1)
    # plt.plot(grids, erf_plot - erf_plot[1])

    plt.show()
