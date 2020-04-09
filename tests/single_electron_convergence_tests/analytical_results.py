import numpy as np
from scipy import special
from scipy import integrate
from scipy import optimize
import math


def exp_hydrogenic(grids, A=1.071295, k=(1. / 2.385345), a=0, Z=1):
    """Exponential potential for 1D Hydrogenic atom.

    A 1D potential which can be used to mimic corresponding 3D
    electronic structure. Similar in form to the soft-Coulomb
    interaction, however there is a cusp occurring at x = 0 for
    a -> 0. Please refer to:

    Thomas E Baker, E Miles Stoudenmire, Lucas O Wagner, Kieron Burke,
    and  Steven  R  White. One-dimensional mimicking of electronic structure:
    The case for exponentials. Physical Review B,91(23):235141, 2015.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      Z: the “charge” felt by an electron from the nucleus.
      A: fitting parameter.
      k: fitting parameter.
      a: fitting parameter used to soften the cusp at the origin.

    Returns:
      vp: Potential on grid.
        (num_grid,)
    """
    vp = -Z * A * np.exp(-k * (grids ** 2 + a ** 2) ** .5)
    return vp


def exp_hydro_cont_well(grids, A, k, d, Z=1, a=0):
    """Exponential potential in continuous well.

    The 1D exp_hydrogenic() potential with a continuous constant potential
    outside the well width, d.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      A: fitting parameter.
      k: fitting parameter.
      a: fitting parameter used to soften the cusp at the origin.
      d: well width
      Z: the “charge” felt by an electron from the nucleus.


    Returns:
      vp: Potential on grid.
        (num_grid,)
    """
    vp = []
    for x in grids:
        if np.abs(x) <= d / 2:
            vp.append(-Z * A * np.exp(-k * (x ** 2 + a ** 2) ** .5))
        else:
            v0 = -Z * A * np.exp(-k * ((d / 2) ** 2 + a ** 2) ** .5)
            vp.append(v0)
    return np.asarray(vp)


def exp_hydro_discont_well(grids, A, k, a, d, Z=1):
    """Exponential potential in continuous well.

    The 1D exp_hydrogenic() potential with a discontinuous vanishing potential
    outside the well width, d.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      A: fitting parameter.
      k: fitting parameter.
      a: fitting parameter used to soften the cusp at the origin.
      d: well width
      Z: the “charge” felt by an electron from the nucleus.


    Returns:
      vp: Potential on grid.
        (num_grid,)
    """
    vp = []
    for x in grids:
        if np.abs(x) <= d / 2:
            vp.append(-Z * A * np.exp(-k * (x ** 2 + a ** 2) ** .5))
        else:
            vp.append(0.)
    return np.asarray(vp)


def z(x, A, k, Z=1):
    """ Function used commonly through analytical the below root calculations

    """
    z = (2 / k) * ((2 * A * Z) ** .5) * np.exp(-(k / 2) * np.abs(x))
    return z


def exp_hydro_open_roots(A, k, Z=1):
    """ Exponential potential in open boundary conditions. The wavefunction
    is set to vanish at +/- infinity. The following returns a list of roots
    {v_i} which can be used to compute analytical bound state energies from:

                        E_i = -(1/8)(k * v_i)^2


    """

    z0 = z(0, A, k, Z)

    jv = lambda x: special.jv(x, z0)
    # derivative of jv
    jv_prime = lambda x: .5 * (special.jv(-1 + x, z0) - special.jv(1 + x, z0))

    lower_bound = 10 * Z
    bracket_list = np.linspace(0, lower_bound, 2 * lower_bound)

    root_list = []
    prev = 0
    for bound in bracket_list:
        try:
            sol = optimize.root_scalar(jv_prime, bracket=[prev, bound])
            root_list.append(sol.root)
        except:
            pass
        try:
            sol = optimize.root_scalar(jv, bracket=[prev, bound])
            root_list.append(sol.root)
        except:
            pass

        prev = bound

    root_list.sort(reverse=True)

    return root_list


def exp_hydro_roots(A, k, L, Z=1):
    """ Exponential potential in finite boundary conditions. The wavefunction
    is set to vanish at +/- L/2. The following returns a list of roots
    {v_i} which can be used to compute analytical bound state energies from:

                        E_i = -(1/8)(k * v_i)^2

    """
    z0 = z(0, A, k, Z)
    zL = z(L / 2., A, k, Z)

    odd_states = lambda x: special.jv(x, z0) - (
            (special.jv(x, zL) / special.jv(-x, zL)) * special.jv(
        -x, z0))

    even_states = lambda x: special.jvp(x, z0) - (
            (special.jv(x, zL) / special.jv(-x,
                                            zL)) * special.jvp(
        -x, z0))

    lower_bound = 10 * Z
    bracket_list = np.linspace(0, lower_bound + .01, 2 * lower_bound)

    root_list = []
    prev = 0
    for bound in bracket_list:

        try:
            sol = optimize.root_scalar(even_states, bracket=[prev, bound])
            # the special.jv(-v, x) I think has a bug for pure integers
            if np.abs(int(round(sol.root)) - sol.root) > 10 ** -9:
                root_list.append(sol.root)
        except:
            pass

        try:
            sol = optimize.root_scalar(odd_states, bracket=[prev, bound])
            if np.abs(int(round(sol.root)) - sol.root) > 10 ** -9:
                root_list.append(sol.root)
        except:
            pass

        prev = bound

    root_list.sort(reverse=True)

    return root_list


def exp_hydro_cont_well_roots(A, k, d, Z=1):
    """ Exponential potential in continuous well in finite boundary conditions.
    The interior wavefunction within the well is matched to the exponential decaying
    wavefunction outside the well. The total wavefunction and its derivative are therefore
    continuous at the well boundary. The following returns a list of roots
    {v_i} which can be used to compute analytical bound state energies from:

                        E_i = -(1/8)(k * v_i)^2

    """
    z0 = z(0, A, k, Z)
    zL = z(d / 2., A, k, Z)
    V0 = exp_hydrogenic(d / 2, A, k, 0, Z)
    u = lambda x: (np.abs(2 * (((1 / 8) * (x ** 2) * (k ** 2)) + V0))) ** .5

    C2 = lambda x: (special.jvp(x, zL) - (2 * u(x) / (k * zL)) * special.jv(x,
                                                                            zL)) * np.exp(
        -1 * u(x) * (d / 2)) / (special.jv(-x, zL) * special.jvp(x,
                                                                 zL) - special.jvp(
        -x,
        zL) * special.jv(
        x,
        zL))

    C1 = lambda x: (np.exp(-1 * u(x) * (d / 2)) - C2(x) * special.jv(-x,
                                                                     zL)) / special.jv(
        x, zL)

    even_states = lambda x: C1(x) * special.jvp(x, z0) + C2(x) * special.jvp(-x,
                                                                             z0)
    odd_states = lambda x: C1(x) * special.jv(x, z0) + C2(x) * special.jv(-x,
                                                                          z0)

    lower_bound = 10 * Z
    bracket_list = np.linspace(0, lower_bound + .01, 2 * lower_bound)

    root_list = []
    prev = 0
    for bound in bracket_list:

        try:
            sol = optimize.root_scalar(even_states, bracket=[prev, bound])
            # the special.jv(-v, x) I think has a bug for pure integers
            if np.abs(int(round(sol.root)) - sol.root) > 10 ** -9:
                root_list.append(sol.root)
        except:
            pass

        try:
            sol = optimize.root_scalar(odd_states, bracket=[prev, bound])
            if np.abs(int(round(sol.root)) - sol.root) > 10 ** -9:
                root_list.append(sol.root)
        except:
            pass

        prev = bound

    root_list.sort(reverse=True)

    return root_list


def exp_hydro_discont_well_roots(A, k, d, Z=1):
    """ Exponential potential in discontinuous well in finite boundary conditions.
    The interior wavefunction within the well is matched to the exponential decaying
    wavefunction outside the well. The total wavefunction and its derivative are therefore
    continuous at the well boundary. The following returns a list of roots
    {v_i} which can be used to compute analytical bound state energies from:

                           E_i = -(1/8)(k * v_i)^2

    """
    z0 = z(0, A, k, Z)
    zL = z(d / 2., A, k, Z)

    C2 = lambda x: (special.jvp(x, zL) - (x / zL) * special.jv(x, zL)) * np.exp(
        -(k * x / 2) * d / 2) / (
                           special.jv(-x, zL) * special.jvp(x,
                                                            zL) - special.jvp(
                       -x,
                       zL) * special.jv(x,
                                        zL))

    C1 = lambda x: (np.exp(-(k * x / 2) * d / 2) - C2(x) * special.jv(-x,
                                                                      zL)) / special.jv(
        x, zL)

    even_states = lambda x: C1(x) * special.jvp(x, z0) + C2(x) * special.jvp(-x,
                                                                             z0)
    odd_states = lambda x: C1(x) * special.jv(x, z0) + C2(x) * special.jv(-x,
                                                                          z0)

    lower_bound = 10 * Z
    bracket_list = np.linspace(0, lower_bound + .01, 2 * lower_bound)

    root_list = []
    prev = 0
    for bound in bracket_list:

        try:
            sol = optimize.root_scalar(even_states, bracket=[prev, bound])
            # the special.jv(-v, x) I think has a bug for pure integers
            if np.abs(int(round(sol.root)) - sol.root) > 10 ** -9:
                root_list.append(sol.root)
        except:
            pass

        try:
            sol = optimize.root_scalar(odd_states, bracket=[prev, bound])
            if np.abs(int(round(sol.root)) - sol.root) > 10 ** -9:
                root_list.append(sol.root)
        except:
            pass

        prev = bound

    root_list.sort(reverse=True)

    return root_list
