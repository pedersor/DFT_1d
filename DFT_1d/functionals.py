"""
Functionals
###########

**Summary** 
    Defined grid-based exchange-correlation functionals, fock operator,
    and potentials for 1D systems.

.. moduleauthor::
    `Ryan Pederson <pedersor@uci.edu>`_ ORCID: `0000-0002-7228-9478 <https://orcid.org/0000-0002-7228-9478>`_,
    `Li Li`,
    `Chris (Jielun) Chen`,
    `Johnny Kozlowski`

.. todo::
    * extensive renaming/reorganizing.
    * hf: replace with Chris' matrix muliplication code. Cleaner & Jittable.
    * Docs need love.
"""

import functools

import jax
import jax.numpy as jnp
from jax.config import config
import numpy as np

from DFT_1d import ext_potentials
from DFT_1d import constants
from DFT_1d.utils import get_dx

# Set the default dtype as float64
config.update('jax_enable_x64', True)


def v_ee(
    grids,
    A=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
    k=constants.EXPONENTIAL_COULOMB_KAPPA,
):
  vp = A * jnp.exp(-k * jnp.abs(grids))
  return vp


@functools.partial(jax.jit, static_argnums=(2,))
def _get_hartree_potential(grids, n, v_ee):
  dx = grids[1] - grids[0]
  n1 = jnp.expand_dims(n, axis=0)
  r1 = jnp.expand_dims(grids, axis=0)
  r2 = jnp.expand_dims(grids, axis=1)
  return jnp.sum(n1 * v_ee(r1 - r2), axis=1) * dx


def get_hartree_potential(grids, n, v_ee=v_ee):
  return _get_hartree_potential(grids, n, v_ee)


class BaseHartreeFock:

  def __init__(self, grids):
    self.grids = grids
    self.dx = get_dx(grids)

  def v_hf(self, grids, n, v_ext):
    """Total HF potential, v_{eff}."""
    v_h = self.hartree_potential()
    return v_ext(grids) + v_h(grids=grids, n=n)

  def hartree_potential(self):
    return NotImplementedError()


class ExponentialHF(BaseHartreeFock):

  def __init__(self,
               grids,
               A=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
               k=constants.EXPONENTIAL_COULOMB_KAPPA):
    self.grids = grids
    self.num_grids = len(grids)
    self.A = A
    self.k = k
    self.dx = get_dx(grids)

  def hartree_potential(self):
    return get_hartree_potential

  def update_fock_matrix(self, wave_function):
    # fock matrix will be implemented as fock operator,
    # see RP logbook 9/6/19

    num_electrons = len(wave_function)
    mat = np.zeros((self.num_grids, self.num_grids))

    for j in range(num_electrons):

      mat_j = np.zeros((self.num_grids, self.num_grids))
      for row in range(self.num_grids):
        for column in range(self.num_grids):
          mat_j[row, column] = ext_potentials.exp_hydrogenic(
              self.grids[row] - self.grids[column], self.A,
              self.k) * \
                               wave_function[j][column] * \
                               wave_function[j][row] * self.dx

      mat += mat_j

    return mat

  def get_exchange_energy(self, wave_function):
    """Obtain exchange energy 'exactly' from double integral over HF
        orbitals """
    num_electrons = len(wave_function)

    E_x = 0
    for i in range(num_electrons):
      outer_int_tot = 0
      for j in range(num_electrons):
        inner_int_tot = 0
        for x_i, x in enumerate(self.grids):
          int_fn_of_x = np.zeros(self.num_grids)
          for x_prime_i, x_prime in enumerate(self.grids):
            int_fn_of_x[x_i] += -1 * ext_potentials.exp_hydrogenic(
                x - x_prime, self.A, self.k) * \
                                wave_function[i][x_i] * \
                                wave_function[j][x_i] * \
                                wave_function[i][
                                    x_prime_i] * \
                                wave_function[j][
                                    x_prime_i] * self.dx

          inner_int_tot += int_fn_of_x[x_i] * self.dx
        outer_int_tot += inner_int_tot
      E_x += outer_int_tot

    E_x = -.5 * E_x
    return E_x


class BaseExchangeCorrelationFunctional:

  def __init__(self, grids):
    self.grids = grids
    self.dx = get_dx(grids)

  def hartree_potential(self):
    """Hartree potential."""
    return NotImplementedError()

  def get_hartree_energy(self, n):
    """Hartree energy."""
    return NotImplementedError()

  def get_xc_potential_up(self, n_up, n_down):
    raise NotImplementedError()

  def get_xc_potential_down(self, n_up, n_down):
    raise NotImplementedError()

  def get_xc_potential(self, n):
    """Spin-free XC potential."""
    raise NotImplementedError()

  def get_ks_potential_up(self, grids, v_ext, n_up, n_down):
    """Spin-decomposed 'up' KS potential."""
    v_h = self.hartree_potential()
    n = n_up + n_down

    return (v_ext(grids) + v_h(grids=grids, n=n) +
            self.get_xc_potential_up(n_up, n_down))

  def get_ks_potential_down(self, grids, v_ext, n_up, n_down):
    """Spin-decomposed 'down' KS potential."""
    v_h = self.hartree_potential()
    n = n_up + n_down

    return (v_ext(grids) + v_h(grids=grids, n=n) +
            self.get_xc_potential_down(n_up, n_down))

  def get_ks_potential(self, grids, n, v_ext):
    """Spin-free KS potential."""
    v_h = self.hartree_potential()
    return (v_ext(grids) + v_h(grids=grids, n=n) + self.get_xc_potential(n))

  def get_xc_energy(self, n, *args):
    raise NotImplementedError()


class ExponentialLSDFunctional(BaseExchangeCorrelationFunctional):
  """local density approximation (LDA) for exponentially repelling electrons.
    For more details see [Baker2015]_.
    """

  def __init__(self,
               grids,
               A=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
               k=constants.EXPONENTIAL_COULOMB_KAPPA):
    super(ExponentialLSDFunctional, self).__init__(grids=grids)
    self.A = A
    self.k = k

  def hartree_potential(self):
    return get_hartree_potential

  def get_hartree_energy(self, n):
    v_h = self.hartree_potential()
    return 0.5 * jnp.dot(v_h(grids=self.grids, n=n), n) * self.dx

  def exchange_energy_density(self, n, zeta):
    """Exchange energy density."""
    y = jnp.pi * n / self.k
    e_x = self.A * self.k * (
        jnp.log(1 + (y**2) * ((1 + zeta)**2)) - 2 * y *
        (1 + zeta) * jnp.arctan(y * (1 + zeta)) + jnp.log(1 + (y**2) * (
            (-1 + zeta)**2)) - 2 * y *
        (-1 + zeta) * jnp.arctan(y * (-1 + zeta))) / (4 * (jnp.pi**2))

    return jnp.nan_to_num(e_x / n)

  def correlation_energy_density(self, n, zeta):
    """Correlation energy density. Parameters derived in [Baker2015]_."""

    def correlation_expression(n, alpha, beta, gamma, delta, eta, sigma, nu):
      """
        Pade approximate with parameters.
        """
      y = jnp.pi * n / self.k
      return (-self.A * self.k * (y**2) /
              (jnp.pi**2)) / (alpha + beta *
                              (y**(1. / 2.)) + gamma * y + delta *
                              (y**(3. / 2.)) + eta * (y**2) + sigma *
                              (y**(5. / 2.)) + nu *
                              (jnp.pi * (self.k**2) / self.A) * (y**3))

    # Parameters below derived in [Baker2015]_.
    unpol = correlation_expression(n, 2, -1.00077, 6.26099, -11.9041, 9.62614,
                                   -1.48334, 1)
    pol = correlation_expression(n, 180.891, -541.124, 651.615, -356.504,
                                 88.0733, -4.32708, 8)
    e_c = unpol + (zeta**2) * (pol - unpol)
    return jnp.nan_to_num(e_c / n)

  def xc_energy_density(self, n, zeta):
    return (self.exchange_energy_density(n, zeta) +
            self.correlation_energy_density(n, zeta))

  def get_exchange_energy(self, n_up, n_down):
    n = n_up + n_down
    zeta = (n_up - n_down) / n

    return jnp.dot(self.exchange_energy_density(n, zeta), n) * self.dx

  def get_correlation_energy(self, n_up, n_down):
    n = n_up + n_down
    zeta = (n_up - n_down) / n

    return jnp.dot(self.correlation_energy_density(n, zeta), n) * self.dx

  def get_xc_energy(self, n_up, n_down):
    n = n_up + n_down
    zeta = (n_up - n_down) / n

    return jnp.dot(self.xc_energy_density(n, zeta), n) * self.dx

  def get_exchange_potential(self, n_up, n_down):
    x_potential_up = jax.grad(self.get_exchange_energy, argnums=0)(
        n_up, n_down) / self.dx

    x_potential_down = jax.grad(self.get_exchange_energy, argnums=1)(
        n_up, n_down) / self.dx

    return x_potential_up, x_potential_down

  def get_correlation_potential(self, n_up, n_down):
    c_potential_up = jax.grad(self.get_correlation_energy, argnums=0)(
        n_up, n_down) / self.dx

    c_potential_down = jax.grad(self.get_correlation_energy, argnums=1)(
        n_up, n_down) / self.dx

    return c_potential_up, c_potential_down

  def get_xc_potential_up(self, n_up, n_down):
    xc_potential_up = jax.grad(self.get_xc_energy, argnums=0)(n_up,
                                                              n_down) / self.dx

    return xc_potential_up

  def get_xc_potential_down(self, n_up, n_down):
    xc_potential_down = jax.grad(self.get_xc_energy, argnums=1)(
        n_up, n_down) / self.dx
    return xc_potential_down


class AnalyticalLSD(ExponentialLSDFunctional):
  """local density approximation (LDA) for exponentially repelling electrons.
  For more details see [Baker2015]_. Here we use analytical expressions only
  and do not require automatic differentiation or JAX.
  """

  def __init__(self, grids):
    super(AnalyticalLSD, self).__init__(grids=grids)

  def _set_pade_approx_params(self, n):
    """Set Pade approximation parameters. They are derived in [Baker2015]_.

    Args:
        n: system density on a grid.
    Returns:
        u1, p1, u2, p2: parameters to be used.
    """

    def expression_1(n, alpha, beta, gamma, delta, eta, sigma, nu):
      y = np.pi * n / self.k
      return alpha + beta * (y**(1. / 2.)) + gamma * y + delta * (y**(
          3. / 2.)) + eta * (y**2) + sigma * (y**(5. / 2.)) + nu * (
              np.pi * (self.k**2) / self.A) * (y**3)

    def expression_2(n, alpha, beta, gamma, delta, eta, sigma, nu):
      return (6 * (n**3) * (np.pi**4) * self.k * nu + self.A * np.sqrt(np.pi) *
              (4 * (n**2) * (np.pi**(3. / 2.)) * eta +
               2 * n * np.sqrt(np.pi) * gamma * self.k + 3 * n * np.pi * delta *
               np.sqrt(n / self.k) * self.k + beta * np.sqrt(n / self.k) *
               (self.k**2) + 5 * n * (np.pi**2) *
               ((n / self.k)**(3. / 2.)) * self.k * sigma)) / (2 * self.A * n *
                                                               (self.k**2))

    u1 = expression_1(n, 2, -1.00077, 6.26099, -11.9041, 9.62614, -1.48334, 1)
    p1 = expression_1(n, 180.891, -541.124, 651.615, -356.504, 88.0733,
                      -4.32708, 8)
    u2 = expression_2(n, 2, -1.00077, 6.26099, -11.9041, 9.62614, -1.48334, 1)
    p2 = expression_2(n, 180.891, -541.124, 651.615, -356.504, 88.0733,
                      -4.32708, 8)

    return u1, p1, u2, p2

  def get_xc_potential_up(self, n_up, n_down):
    """Exchange-Correlation Potential for up electrons,
    :math:`v_{xc, \\uparrow} = d/dn_{\\uparrow} e_{xc}`.

    Args:
        n: system density on a grid.
        n_up: up spin density on a grid.
        n_down: down spin density on a grid.
    Returns:
        `ndarray`: the up XC potential on a grid.
    """

    n = n_up + n_down

    pi = np.pi
    u1, p1, u2, p2 = self._set_pade_approx_params(n)

    v_x = -(self.A / (pi)) * (np.arctan(2 * pi * n_up / self.k))
    v_c = (self.A * (2 * p1 * (u1**2) * (n_down - n_up) + (u1**2) *
                     ((n_down - n_up)**2) * p2 - 4 * (p1**2) * n_down *
                     (u1 - n_up * u2))) / ((p1**2) * (u1**2) * self.k)
    return v_x + v_c

  def get_xc_potential_down(self, n_up, n_down):
    """Exchange-Correlation Potential for up electrons,
    :math:`v_{xc, \\downarrow} = d/dn_{\\downarrow} e_{xc}`.

    Args:
        n: system density on a grid.
        n_up: up spin density on a grid.
        n_down: down spin density on a grid.
    Returns:
        `ndarray`: the down XC potential on a grid.
    """
    n = n_up + n_down

    pi = np.pi
    u1, p1, u2, p2 = self._set_pade_approx_params(n)

    v_x = -(self.A / (pi)) * (np.arctan(2 * pi * n_down / self.k))
    v_c = (self.A * (2 * p1 * (u1**2) * (-n_down + n_up) + (u1**2) *
                     ((n_down - n_up)**2) * p2 - 4 * (p1**2) * n_up *
                     (u1 - n_down * u2))) / ((p1**2) * (u1**2) * self.k)
    return v_x + v_c


class ExponentialLDAFunctional(BaseExchangeCorrelationFunctional):

  def __init__(self,
               grids,
               A=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
               k=constants.EXPONENTIAL_COULOMB_KAPPA):
    super(ExponentialLDAFunctional, self).__init__(grids=grids)

  def hartree_potential(self):
    return get_hartree_potential

  def get_hartree_energy(self, n):
    v_h = self.hartree_potential()
    return 0.5 * jnp.dot(v_h(grids=self.grids, n=n), n) * self.dx

  def e_x(self, n):
    return self.exchange_energy_density(n) * n

  def e_c(self, n):
    return self.correlation_energy_density(n) * n

  def exchange_energy_density(self,
                              density,
                              amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
                              kappa=constants.EXPONENTIAL_COULOMB_KAPPA,
                              epsilon=1e-15):
    """Exchange energy density for uniform gas with exponential coulomb.

    Equation 17 in the following paper provides the exchange energy per length
    for 1d uniform gas with exponential coulomb interaction.

    One-dimensional mimicking of electronic structure: The case for exponentials.
    Physical Review B 91.23 (2015): 235141.
    https://arxiv.org/pdf/1504.05620.pdf

    y = pi * density / kappa
    exchange energy per length
        = amplitude * kappa * (ln(1 + y ** 2) - 2 * y * arctan(y)) / (2 * pi ** 2)

    exchange energy density
        = exchange energy per length * pi / (kappa * y)
        = amplitude / (2 * pi) * (ln(1 + y ** 2) / y - 2 * arctan(y))

    Dividing by y may cause numerical instability when y is close to zero. Small
    value epsilon is introduced to prevent it.

    When density is smaller than epsilon, the exchange energy density is computed
    by its series expansion at y=0:

    exchange energy density = amplitude / (2 * pi) * (-y + y ** 3 / 6)

    Note the exchange energy density converge to constant -amplitude / 2 at high
    density limit.

    Args:
      density: Float numpy array with shape (num_grids,).
      amplitude: Float, parameter of exponential Coulomb interaction.
      kappa: Float, parameter of exponential Coulomb interaction.
      epsilon: Float, a constant for numerical stability.

    Returns:
      Float numpy array with shape (num_grids,).
    """
    y = jnp.pi * density / kappa
    return jnp.where(
        y > epsilon,
        amplitude / (2 * jnp.pi) * (jnp.log(1 + y**2) / y - 2 * jnp.arctan(y)),
        amplitude / (2 * jnp.pi) * (-y + y**3 / 6))

  def correlation_energy_density(
      self,
      density,
      amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
      kappa=constants.EXPONENTIAL_COULOMB_KAPPA):
    """Exchange energy density for uniform gas with exponential coulomb.

    Equation 24 in the following paper provides the correlation energy per length
    for 1d uniform gas with exponential coulomb interaction.

    One-dimensional mimicking of electronic structure: The case for exponentials.
    Physical Review B 91.23 (2015): 235141.
    https://arxiv.org/pdf/1504.05620.pdf

    y = pi * density / kappa
    correlation energy per length
        = -amplitude * kappa * y ** 2 / (pi ** 2) / (
          alpha + beta * sqrt(y) + gamma * y + delta * sqrt(y ** 3)
          + eta * y ** 2 + sigma * sqrt(y ** 5)
          + nu * pi * kappa ** 2 / amplitude * y ** 3)

    correlation energy density
        = correlation energy per length * pi / (kappa * y)
        = -amplitude * y / pi / (
          alpha + beta * sqrt(y) + gamma * y + delta * sqrt(y ** 3)
          + eta * y ** 2 + sigma * sqrt(y ** 5)
          + nu * pi * kappa ** 2 / amplitude * y ** 3)

    Note the correlation energy density converge to zero at high density limit.

    Args:
      density: Float numpy array with shape (num_grids,).
      amplitude: Float, parameter of exponential Coulomb interaction.
      kappa: Float, parameter of exponential Coulomb interaction.

    Returns:
      Float numpy array with shape (num_grids,).
    """
    y = jnp.pi * density / kappa
    alpha = 2.
    beta = -1.00077
    gamma = 6.26099
    delta = -11.9041
    eta = 9.62614
    sigma = -1.48334
    nu = 1.
    # The derivative of sqrt is not defined at y=0, we use two jnp.where to avoid
    # nan at 0.
    finite_y = jnp.where(y == 0., 1., y)
    out = -amplitude * finite_y / jnp.pi / (
        alpha + beta * jnp.sqrt(finite_y) + gamma * finite_y +
        delta * finite_y**1.5 + eta * finite_y**2 + sigma * finite_y**2.5 +
        nu * jnp.pi * kappa**2 / amplitude * finite_y**3)
    return jnp.where(y == 0., -amplitude * y / jnp.pi / alpha, out)

  def xc_energy_density(self, density):
    """XC energy density of Local Density Approximation with exponential coulomb.

    One-dimensional mimicking of electronic structure: The case for exponentials.
    Physical Review B 91.23 (2015): 235141.
    https://arxiv.org/pdf/1504.05620.pdf

    Args:
      density: Float numpy array with shape (num_grids,).

    Returns:
      Float numpy array with shape (num_grids,).
    """
    return (self.exchange_energy_density(density) +
            self.correlation_energy_density(density))

  def get_exchange_energy(self, density):
    return jnp.dot(self.exchange_energy_density(density), density) * self.dx

  def get_correlation_energy(self, density):
    return jnp.dot(self.correlation_energy_density(density), density) * self.dx

  def get_xc_energy(self, density):
    r"""Gets xc energy.

    E_xc = \int density * xc_energy_density_fn(density) dx.

    Args:
      density: Float numpy array with shape (num_grids,).

    Returns:
      Float.
    """
    return jnp.dot(self.xc_energy_density(density), density) * self.dx

  def get_xc_potential(self, density):
    """Gets xc potential.

    The xc potential is derived from xc_energy_density through automatic
    differentiation.

    Args:
      density: Float numpy array with shape (num_grids,).

    Returns:
      Float numpy array with shape (num_grids,).
    """
    return jnp.nan_to_num(jax.grad(self.get_xc_energy)(density) / self.dx)
