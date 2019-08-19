import numpy as np
from scipy import special
from scipy import integrate
from scipy import optimize
import math


def gaussian_dips(grids, coeff, sigma, mu):
    """Potential of sum of Gaussian dips.

    The i-th Gaussian dip is
      -coeff[i] * np.exp(-(grids - mu[i]) ** 2 / (2 * sigma[i] ** 2))

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      coeff: numpy array of coefficient for each gaussian dip.
        (num_dips,)
      sigma: numpy array of standard deviation for each gaussian dip.
        (num_dips,)
      mu: numpy array of mean for each gaussian dip.
        (num_dips,)

    Returns:
      vp: Potential on grid.
        (num_grids,)
    """
    grids = np.expand_dims(grids, axis=0)
    coeff = np.expand_dims(coeff, axis=1)
    sigma = np.expand_dims(sigma, axis=1)
    mu = np.expand_dims(mu, axis=1)

    vps = -coeff * np.exp(-(grids - mu) ** 2 / (2 * sigma ** 2))
    vp = np.sum(vps, axis=0)
    return vp


def harmonic_oscillator(grids, k=1.):
    """Potential of quantum harmonic oscillator.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      k: strength constant for potential vp = 0.5 * k * grids ** 2

    Returns:
      vp: Potential on grid.
        (num_grid,)
    """
    vp = 0.5 * k * grids ** 2
    return vp


def kronig_penney(grids, a, b, v0):
    """Kronig-Penney model potential. For more information, see:

    https://en.wikipedia.org/wiki/Particle_in_a_one-dimensional_lattice#Kronig%E2%80%93Penney_model

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      a: periodicity of 1d lattice
      b: width of potential well
      v0: negative float. It is the depth of potential well.

    Returns:
      vp: Potential on grid.
        (num_grid,)
    """
    if v0 >= 0:
        raise ValueError('v0 is expected to be negative but got %4.2f.' % v0)
    if b >= a:
        raise ValueError('b is expected to be less than a but got %4.2f.' % b)

    vp = []
    for x in grids:
        if x < (a - b):
            vp.append(0.)
        else:
            vp.append(v0)

    return np.asarray(vp)


def exp_hydrogenic(grids, A, k, a, Z=1):
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

    even_states = lambda x: special.jvp(x, z0) - ((special.jv(x, zL) / special.jv(-x,
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

    C2 = lambda x: (special.jvp(x, zL) - (2 * u(x) / (k * zL)) * special.jv(x, zL)) * np.exp(
        -1 * u(x) * (d / 2)) / (special.jv(-x, zL) * special.jvp(x, zL) - special.jvp(-x,
                                                                                      zL) * special.jv(
        x,
        zL))

    C1 = lambda x: (np.exp(-1 * u(x) * (d / 2)) - C2(x) * special.jv(-x, zL)) / special.jv(x, zL)

    even_states = lambda x: C1(x) * special.jvp(x, z0) + C2(x) * special.jvp(-x, z0)
    odd_states = lambda x: C1(x) * special.jv(x, z0) + C2(x) * special.jv(-x, z0)

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
                           special.jv(-x, zL) * special.jvp(x, zL) - special.jvp(-x,
                                                                                 zL) * special.jv(x,
                                                                                                  zL))

    C1 = lambda x: (np.exp(-(k * x / 2) * d / 2) - C2(x) * special.jv(-x, zL)) / special.jv(x, zL)

    even_states = lambda x: C1(x) * special.jvp(x, z0) + C2(x) * special.jvp(-x, z0)
    odd_states = lambda x: C1(x) * special.jv(x, z0) + C2(x) * special.jv(-x, z0)

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


def exp_H2plus(grids, A, k, a, d, Z=1):
    """ Two Exponential potentials separated by a distance d.


    """
    vp = exp_hydrogenic(grids - d / 2, A, k, a, Z=1) + exp_hydrogenic(grids + d / 2, A, k, a, Z=1)
    return vp


def poschl_teller(grids, lam, a=1., center=0.):
    r"""Poschl-Teller potential.

    Poschl-Teller potential is a special class of potentials for which the
    one-dimensional Schrodinger equation can be solved in terms of Special
    functions.

    https://en.wikipedia.org/wiki/P%C3%B6schl%E2%80%93Teller_potential

    The general form of the potential is

    v(x) = -\frac{\lambda(\lambda + 1)}{2} a^2 \frac{1}{\cosh^2(a x)}

    It holds M=ceil(\lambda) levels, where \lambda is a positive float.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      lam: float, lambda in the Poschl-Teller potential function.
      a: float, coefficient in the Poschl-Teller potential function.
      center: float, the center of the potential.

    Returns:
      Potential on grid with shape (num_grid,)

    Raises:
      ValueError: If lam is not positive.
    """
    if lam <= 0:
        raise ValueError('lam is expected to be positive but got %4.2f.' % lam)
    return -lam * (lam + 1) * a ** 2 / (2 * np.cosh(a * (grids - center)) ** 2)


def _valid_poschl_teller_level_lambda(level, lam):
    """Checks whether level and lambda is valid.

    Args:
      level: positive integer, the ground state is level=1.
      lam: positive float, lambda.

    Raises:
      ValueError: If lam is not positive; level is less than 1 or level is greater
        than the total number of levels the potential can hold.
    """
    if lam <= 0:
        raise ValueError('lam is expected to be positive but got %4.2f.' % lam)
    level = int(level)
    if level < 1:
        raise ValueError(
            'level is expected to be greater or equal to 1, but got %d.' % level)
    if level > np.ceil(lam):
        raise ValueError(
            'lam %4.2f can hold %d levels, but got level %d.'
            % (lam, np.ceil(lam), level))


def poschl_teller_energy(level, lam, a=1.):
    """Analytic solution of the total energy filled up to level-th eigenstate.

    The solution can be found in second row of Table 1 in

    Leading corrections to local approximations. II. The case with turning points
    Raphael F. Ribeiro and Kieron Burke, Phys. Rev. B 95, 115115
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.115115

    Args:
      level: positive integer, the ground state is level=1.
      lam: positive float, lambda.
      a: float, coefficient in Poschl-Teller potential.

    Returns:
      Float, the total energy from first to the level-th eigenstate.
    """
    total_energy = 0.
    for i in range(1, int(level) + 1):
        total_energy += poschl_teller_eigen_energy(i, lam, a)
    return total_energy


def poschl_teller_eigen_energy(level, lam, a=1.):
    """Analytic solution of the level-th eigen energy for Poschl-Teller potential.

    This is the energy level for Poschl-Teller potential with float lambda. The
    solution can be found in second row of Table 1 in

    Leading corrections to local approximations. II. The case with turning points
    Raphael F. Ribeiro and Kieron Burke, Phys. Rev. B 95, 115115
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.115115

    Args:
      level: positive integer, the ground state is level=1.
      lam: positive float, lambda.
      a: float, coefficient in Poschl-Teller potential.

    Returns:
      Float, the energy of the level-th eigenstate.
    """
    level = int(level)
    _valid_poschl_teller_level_lambda(level, lam)
    a2 = a ** 2  # a square
    return -a2 * (np.sqrt(lam * (lam + 1) / a2 + 0.25) - level + 0.5) ** 2 / 2
