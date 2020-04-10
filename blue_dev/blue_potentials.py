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


def blue_helium_1d(grids, r0, Z=2, lam=1):
    return ext_potentials.exp_hydrogenic(grids,
                                         Z=Z) - lam * ext_potentials.exp_hydrogenic(
        grids - r0)


def helium(grids, Z=2):
    return -Z / (np.abs(grids))


def blue_H2_1d(grids, pot, r0, lam=1):
    # pot is the pre-generated external potential of the separated H2
    return pot - lam * ext_potentials.exp_hydrogenic(grids - r0)
