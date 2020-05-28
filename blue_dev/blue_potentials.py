import ext_potentials
import numpy as np
from scipy import special
from scipy import integrate
from scipy import optimize
import math


def blue_helium(grids, r0, Z=2, lam=1):
    vp = []
    for r in grids:
        if np.abs(r) < r0:
            vp.append((-Z / np.abs(r)) + (lam / r0))
        else:
            vp.append((-Z + lam) / np.abs(r))

    return np.asarray(vp)


def blue_helium_spherical_erf(grids, r0, gam=1, Z=2, lam=1):
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


def blue_helium_1d_erf(grids, r0, gam=1, Z=2):
    # 1/(2|r-r'|) for r ~ r' and 1/|r-r'| for large separations of r and r'.
    # Decrease/increase 'transition time' using gamma.

    return ext_potentials.exp_hydrogenic(grids,
                                         Z=Z) - 0.5 * ext_potentials.exp_hydrogenic(
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
