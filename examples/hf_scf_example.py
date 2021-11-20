"""
.. _hf_scf_example:

Hartree-Fock Self-Consistent Field Example
##########################################

Summary:
    Generates Hartree-Fock (HF) a tabel from the [Baker2015]_ paper which
    includes Li atom and Li, Be, He, and H atoms.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import functools

from DFT_1d import hf_scf
from DFT_1d import functionals
from DFT_1d import ext_potentials


def hf_scf_atom(grids, num_electrons, num_unpaired_electrons, nuclear_charge):
  """ Example HF-SCF calculation for a 1D atom with
        exponential interactions, see ext_potentials.exp_hydrogenic.

    Args:
        grids: grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
        num_electrons: the number of electrons in the atom.
        nuclear_charge: the nuclear charge Z of the atom.

    Returns:
        HF-SCF solver class.
    """

  v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=nuclear_charge)
  exponential_hf = functionals.ExponentialHF
  solver = hf_scf.HF_Solver(grids,
                            v_ext=v_ext,
                            hf=exponential_hf,
                            num_electrons=num_electrons,
                            num_unpaired_electrons=num_unpaired_electrons)
  solver.solve_self_consistent_density(verbose=1,
                                       energy_converge_tolerance=1e-4)
  return solver


def get_hf_energies(solver):

  if solver.is_converged():
    print()
    print('Converged results:')
  else:
    print()
    warnings.warn('results are not converged!')

  # Non-Interacting Kinetic Energy
  print("T_s =", solver.hf_kinetic_energy)

  # External Potential Energy
  print("V =", solver.ext_potential_energy)

  # Hartree Energy
  print("U =", solver.hartree_energy)

  # Exchange Energy
  print("E_x =", solver.exchange_energy)

  # Total Energy
  print("E =", solver.total_energy)

  return solver


if __name__ == '__main__':
  """ Li atom HF calculation example. """
  h = 0.08
  grids = np.arange(-256, 257) * h
  nuclear_charge = 3
  num_electrons = 3
  num_unpaired_electrons = 1

  hf_solver = hf_scf_atom(grids, num_electrons, num_unpaired_electrons,
                          nuclear_charge)
  get_hf_energies(hf_solver)

  # plot example self-consistent HF density
  plt.plot(grids, hf_solver.density)
  plt.ylabel('$n(x)$', fontsize=16)
  plt.xlabel('$x$', fontsize=16)
  plt.grid(alpha=0.4)
  plt.show()
