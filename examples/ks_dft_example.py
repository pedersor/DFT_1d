"""

.. _ks_dft_example:

Kohn-Sham DFT Example
#####################

Summary:
    Generates LDA DFT values from a tabel from the [Baker2015]_ paper which includes Li atom and Li, Be, He, and H atoms.
"""

import sys
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import functools

from DFT_1d import ks_dft
from DFT_1d import functionals
from DFT_1d import ext_potentials


def lsda_ks_dft_atom(
    grids, num_electrons, num_unpaired_electrons, nuclear_charge):
    """local spin density approximation (LSD) KS-DFT calculation for a 1D atom
    with exponential interactions, see ext_potentials.exp_hydrogenic.

    Args:
        grids: grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
        num_electrons: the number of electrons in the atom.
        nuclear_charge: the nuclear charge Z of the atom.

    Returns:
        KS-DFT solver class.
    """

    v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=nuclear_charge)
    lsd_xc = functionals.ExponentialLSDFunctional
    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, xc=lsd_xc,
                              num_electrons=num_electrons,
                              num_unpaired_electrons=num_unpaired_electrons)
    solver.solve_self_consistent_density(verbose=1)

    return solver


def lda_ks_dft_atom(grids, num_electrons, nuclear_charge):
  """local density approximation (LDA) KS-DFT calculation for a 1D atom with
      exponential interactions, see ext_potentials.exp_hydrogenic.

  Args:
      grids: grids: numpy array of grid points for evaluating 1d potential.
      (num_grids,)
      num_electrons: the number of electrons in the atom.
      nuclear_charge: the nuclear charge Z of the atom.

  Returns:
      KS-DFT solver class.
  """

  v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=nuclear_charge)
  lda_xc = functionals.ExponentialLDAFunctional
  solver = ks_dft.Spinless_KS_Solver(grids, v_ext=v_ext, xc=lda_xc,
                            num_electrons=num_electrons)
  solver.solve_self_consistent_density(verbose=1)

  return solver


def get_ks_dft_energies(solver):

    if solver.is_converged():
      print()
      print('Converged results:')
    else:
      print()
      warnings.warn('results are not converged!')

    # Non-Interacting (Kohn-Sham) Kinetic Energy
    print("T_s =", solver.ks_kinetic_energy)

    # External Potential Energy
    print("V =", solver.ext_potential_energy)

    # Hartree Energy
    print("U =", solver.hartree_energy)

    # Exchange Energy
    print("E_x =", solver.exchange_energy)

    # Correlation Energy
    print("E_c =", solver.correlation_energy)

    # Total Energy
    print("E =", solver.total_energy)

    return solver


if __name__ == '__main__':
    """Li atom L(S)DA calculation example."""
    h = 0.08
    grids = np.arange(-256, 257) * h
    nuclear_charge = 3
    num_electrons = 3
    num_unpaired_electrons = 1

    if num_unpaired_electrons == 0 or num_unpaired_electrons is None:
      ks_solver = lda_ks_dft_atom(grids, num_electrons, nuclear_charge)
      get_ks_dft_energies(ks_solver)
    else:
      ks_solver = lsda_ks_dft_atom(grids, num_electrons, num_unpaired_electrons,
                                  nuclear_charge)
      get_ks_dft_energies(ks_solver)

    # plot example self-consistent LSDA density
    plt.plot(grids, ks_solver.density)
    plt.ylabel('$n(x)$', fontsize=16)
    plt.xlabel('$x$', fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

