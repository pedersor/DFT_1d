"""
Orbital-free (OF) DFT solver
####################

Note(pedersor): Experimental module!
"""

from functools import partial

import numpy as np
from scipy import optimize

from DFT_1d import constants
from DFT_1d import utils


def v_ee(
    grids,
    A=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
    k=constants.EXPONENTIAL_COULOMB_KAPPA,
):
  vp = A * np.exp(-k * np.abs(grids))
  return vp


def get_dx(grids):
  return grids[1] - grids[0]


def get_hartree_energy(grids, density, v_ee_mat):
  n1 = np.expand_dims(density, axis=0)
  n2 = np.expand_dims(density, axis=1)
  return 0.5 * np.sum(n1 * n2 * v_ee_mat) * get_dx(grids)**2


class OFDFT(object):

  def __init__(
      self,
      grids,
      v_ext,
      num_electrons,
      kinetic_en_fn,
      deriv_kinetic_en_fn,
      xc_energy_density_fn=None,
  ):
    self.grids = grids
    self.v_ext = v_ext
    self.num_electrons = num_electrons
    self.kinetic_en_fn = kinetic_en_fn
    self.deriv_kinetic_en_fn = deriv_kinetic_en_fn

    if xc_energy_density_fn is not None:

      self.xc = xc_energy_density_fn

      # faster to cache the Hartree matrix
      r1 = np.expand_dims(grids, axis=0)
      r2 = np.expand_dims(grids, axis=1)
      self.v_ee_mat = v_ee(r1 - r2)

      self.energy_fn = self._energy_fn
      raise NotImplementedError('Hartree and XC energies not implemented yet.')
    else:
      self.energy_fn = self._energy_fn_wo_hxc
      self.grad_energy_fn = self._grad_energy_fn_wo_hxc

    self.optimizer = partial(
        optimize.minimize,
        method='L-BFGS-B',
        options={
            'gtol': 1e-4,
            'maxiter': 500,
        },
    )

  def _energy_fn(self, phi):
    """ Energy with Hartree and XC energies, i.e. 
    
    E = T_s + E_H + E_XC + V_ext 
    
    """

    # get density from phi
    density = phi**2
    density *= self.num_electrons / np.trapz(density, self.grids)

    ke = self.kinetic_en_fn(density=density, grids=self.grids)
    hartree_en = get_hartree_energy(self.grids, density, self.v_ee_mat)
    xc_en = np.trapz(self.xc(density) * density, self.grids)
    ext_en = np.trapz(self.v_ext * density, self.grids)
    tot_en = ke + hartree_en + xc_en + ext_en

    return tot_en

  def _energy_fn_wo_hxc(self, phi):
    """ Energy without Hartree and XC energies, i.e. E = T_s + V_ext """

    # get density from phi
    density = phi**2
    density *= self.num_electrons / np.trapz(density, self.grids)

    ke = self.kinetic_en_fn(density, self.grids)
    ext_en = np.trapz(self.v_ext * density, self.grids)
    tot_en = ke + ext_en

    return tot_en

  def _grad_energy_fn_wo_hxc(self, density):

    d_ts = self.deriv_kinetic_en_fn(density, self.grids)
    d_ext = self.v_ext

    return d_ts + d_ext

  def residual(self, phi):

    scale = self.num_electrons / np.trapz(phi**2, self.grids)
    density = phi**2
    density *= scale

    return 2* (phi*scale) * (self.grad_energy_fn(density) - self.get_mu(density))   

  def get_mu(self, density):

    return np.trapz(
        self.grad_energy_fn(density) * density,
        self.grids,
    ) / self.num_electrons

  def run(self, guess_density=None):

    if guess_density is None:

      def gaussian_density(grids):
        return np.exp(-1 * grids**2)

      guess_density = gaussian_density(self.grids) + 1e-3
      #guess_density = np.where(guess_density < 1e-15, 0, guess_density)
      guess_density *= self.num_electrons / np.trapz(guess_density, self.grids)

    res = self.optimizer(
        fun=self.energy_fn,
        jac=self.residual,
        x0=guess_density**0.5,
    )

    conv_density = res.x**2
    conv_density *= self.num_electrons / np.trapz(conv_density, self.grids)
    en = res.fun

    if not res.success:
      print(f'Warning: {res.message}')
      print()

    return en, conv_density

  def simple_run(self, eta=1e-4, guess_density=None):
      
      if guess_density is None:
  
        def gaussian_density(grids):
          return np.exp(-0.5 * grids**2)
  
        guess_density = gaussian_density(self.grids)
        guess_density *= self.num_electrons / np.trapz(guess_density, self.grids)
  
      phi = guess_density**0.5
      for i in range(4000):
        phi -= eta * self.residual(phi)

        print(self.energy_fn(phi**2))

      density = phi**2
      en = self.energy_fn(density)
      
      return en, density

if __name__ == '__main__':
  # Example using Poschl-Teller potential and Von-Weizsacker kinetic energy

  import matplotlib.pyplot as plt

  def poschl_teller_potential(grids, lam):
    return -(lam * (lam + 1) / 2) / (np.cosh(grids))**2

  h = 0.02
  grids = np.arange(-256, 257) * h
  v_ext = poschl_teller_potential(grids, lam=2)
  #v_ext = 0.5* grids **2 

  # defione kinetic energy functionals to use.
  # note you need to also supply the functional derivative.


  def get_tau_vw(density, grids, num_offset):
    """ von Weizsacker (vW) kinetic energy density."""

    dx = grids[1] - grids[0]
    grad_density = np.gradient(density, dx)
    res = (1 / 8) * (grad_density**2) / (density + num_offset)
    return np.nan_to_num(res)

  def get_t_vw(density, grids, num_offset=1e-3):
    tau = get_tau_vw(density, grids, num_offset)
    t_vw = np.trapz(tau, grids)
    return t_vw

  # second derivative matrix (finite differences)
  d2_mat = utils.DerivativeTool(grids).d2_mat

  def get_deriv_t_vw(density, grids, num_offset=1e-3):
    dx = grids[1] - grids[0]
    grad_density = np.gradient(density, dx)
    grad2_density = d2_mat.dot(density)

    term_1 = (1 / 8) * (grad_density**2 / (density + num_offset)**2)
    term_2 = -(1 / 4) * (grad2_density / (density + num_offset))

    return term_1 + term_2

  ofdft_solver = OFDFT(
      grids,
      v_ext,
      num_electrons=2,
      kinetic_en_fn=get_t_vw,
      deriv_kinetic_en_fn=get_deriv_t_vw,
  )
  # can supply a guess density (default is a gaussian)
  energy, density = ofdft_solver.run(guess_density=None)

  print('energy = ', energy)
  plt.plot(grids, density)
  plt.xlabel('$x$')
  plt.ylabel('$n(x)$')
  plt.savefig('of_dft_density.pdf', bbox_inches='tight')
