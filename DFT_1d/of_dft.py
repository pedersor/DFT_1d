"""
Orbital-free (OF) DFT solver
####################

Note(pedersor): Experimental module!
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import scipy
from jax.config import config

import functionals

config.update("jax_enable_x64", True)


def get_dx(grids):
  return grids[1] - grids[0]


@partial(jax.jit)
def _get_hartree_energy(grids, density, v_ee_mat):
  n1 = jnp.expand_dims(density, axis=0)
  n2 = jnp.expand_dims(density, axis=1)
  return 0.5 * jnp.sum(n1 * n2 * v_ee_mat) * get_dx(grids)**2


def get_hartree_energy(density, grids, v_ee_mat):
  return _get_hartree_energy(density, grids, v_ee_mat)


class OFDFT(object):

  def __init__(
      self,
      grids,
      v_ext,
      num_electrons,
      kinetic_en_fn,
      xc_energy_density_fn=None,
  ):
    self.grids = grids
    self.v_ext = v_ext
    self.num_electrons = num_electrons
    self.kinetic_en_fn = kinetic_en_fn

    if xc_energy_density_fn is not None:

      self.xc = xc_energy_density_fn

      # faster to cache the Hartree matrix
      r1 = jnp.expand_dims(grids, axis=0)
      r2 = jnp.expand_dims(grids, axis=1)
      self.v_ee_mat = functionals.v_ee(r1 - r2)

      self.energy_fn = self._energy_fn
    else:
      self.energy_fn = self._energy_fn_wo_hxc

    self.optimizer = partial(
        scipy.optimize.minimize,
        method='BFGS',
        options={
            'gtol': 1e-4,
            'maxiter': 500,
        },
    )

  @partial(jax.jit, static_argnums=(0,))
  def _energy_fn(self, phi):
    """ Energy with Hartree and XC energies, i.e. 
    
    E = T_s + E_H + E_XC + V_ext 
    
    """

    # get density from phi
    density = phi**2
    density *= self.num_electrons / jnp.trapz(density, self.grids)

    ke = self.kinetic_en_fn(density=density, grids=self.grids)
    hartree_en = get_hartree_energy(self.grids, density, self.v_ee_mat)
    xc_en = jnp.trapz(self.xc(density) * density, self.grids)
    ext_en = jnp.trapz(self.v_ext * density, self.grids)
    tot_en = ke + hartree_en + xc_en + ext_en

    return tot_en

  @partial(jax.jit, static_argnums=(0,))
  def _energy_fn_wo_hxc(self, phi):
    """ Energy without Hartree and XC energies, i.e. E = T_s + V_ext """

    # get density from phi
    density = phi**2
    density *= self.num_electrons / jnp.trapz(density, self.grids)

    ke = self.kinetic_en_fn(density, self.grids)
    ext_en = jnp.trapz(self.v_ext * density, self.grids)
    tot_en = ke + ext_en

    return tot_en

  @partial(jax.jit, static_argnums=(0,))
  def grad_energy_fn(self, phi):
    return jax.grad(self.energy_fn)(phi)

  def np_energy_fn(self, phi):
    return np.array(self.energy_fn(phi))

  def np_grad_energy_fn(self, phi):
    return np.array(self.grad_energy_fn(phi))

  def run(self, guess_density=None):

    if guess_density is None:

      def gaussian_density(grids):
        return np.exp(-0.5 * grids**2)

      guess_density = gaussian_density(self.grids)
      guess_density = np.where(guess_density < 1e-15, 0, guess_density)

    res = self.optimizer(
        fun=self.np_energy_fn,
        jac=self.np_grad_energy_fn,
        x0=guess_density**0.5,
    )

    conv_density = res.x**2
    # normalize
    conv_density *= self.num_electrons / np.trapz(conv_density, self.grids)
    en = res.fun

    if not res.success:
      print(f'Warning: {res.message}')
      print()

    return en, conv_density


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  h = 0.04
  grids = np.arange(-256, 257) * h
  v_ext = 0.5 * grids**2

  def get_tau_vw(density, grids):
    """ von Weizsacker (vW) kinetic energy density."""

    dx = grids[1] - grids[0]
    grad_density = jnp.gradient(density, dx)
    res = (1 / 8) * (grad_density**2) / (density + 1e-10)
    return jnp.nan_to_num(res)

  def get_t_vw(density, grids):
    tau = get_tau_vw(density, grids)
    t_vw = jnp.trapz(tau, grids)
    return t_vw

  ofdft_solver = OFDFT(grids, v_ext, num_electrons=2, kinetic_en_fn=get_t_vw)
  energy, density = ofdft_solver.run(guess_density=None)

  print('energy = ', energy)
  plt.plot(grids, density)
  plt.xlabel('$x$')
  plt.ylabel('$n(x)$')
  plt.savefig('of_dft_density.pdf', bbox_inches='tight')
