"""

.. _ks_dft_example:

Kohn-Sham DFT Example
#####################

Summary:
    Generates LDA DFT values from a tabel from the [Baker2015]_ paper which includes Li atom and Li, Be, He, and H atoms.
"""

import sys
import os

currentpath = os.path.abspath('.')
sys.path.insert(0, os.path.dirname(currentpath))

import ks_dft, functionals, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools


def lsd_ks_dft_atom(grids, num_electrons, nuclear_charge):
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
    lsd_xc = functionals.ExponentialLSDFunctional(grids=grids)
    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, xc=lsd_xc,
                              num_electrons=num_electrons)
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
  lda_xc = functionals.ExponentialLDAFunctional(grids=grids)
  solver = ks_dft.Spinless_KS_Solver(grids, v_ext=v_ext, xc=lda_xc,
                            num_electrons=num_electrons)
  solver.solve_self_consistent_density(verbose=1)

  return solver


def get_latex_table_atoms(grids):
    """Reproduce LSD results in table 2 of [Baker2015]_.

    Args:
        grids: grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)

    Prints:
        copyable latex-formatted table.
    """

    atom_dict = {"H": [1, 1], "He$^+$": [1, 2], "Li$^{2+}$": [1, 3],
                 "Be$^{3+}$": [1, 4], "He": [2, 2], "Li$^+$": [2, 3],
                 "Be$^{2+}$": [2, 4], "Li": [3, 3], "Be$^+$": [3, 4],
                 "Be": [4, 4]}

    # table headers
    print("$N_e$", end=" & ")
    print("Atom/Ion", end=" & ")
    print("$T_s$", end=" & ")
    print("$V$", end=" & ")
    print("$U$", end=" & ")
    print(r"$E^{\text{LDA}}_x$", end=" & ")
    print(r"$E^{\text{LDA}}_c$", end=" & ")
    print(r"$E^{\text{LDA}}$", end=" ")
    print(r'\\')
    print('\hline')

    for key in atom_dict.keys():
        print(atom_dict[key][0], end=" & ")
        print(key, end=" & ")

        solver = lsd_ks_dft_atom(grids, atom_dict[key][0], atom_dict[key][1])
        print(str(round(solver.T_s, 3)), end=" & ")
        print(str(round(solver.V, 3)), end=" & ")
        print(str(round(solver.U, 3)), end=" & ")
        print(str(round(solver.E_x, 3)), end=" & ")
        print(str(round(solver.E_c, 3)), end=" & ")
        print(str(round(solver.E_tot, 3)), end=" ")

        print(r'\\')
        print('\hline')


def get_ks_dft_energies(solver):

    # Non-Interacting (Kohn-Sham) Kinetic Energy
    print("T_s =", solver.T_s)

    # External Potential Energy
    print("V =", solver.V)

    # Hartree Energy
    print("U =", solver.U)

    # Exchange Energy
    print("E_x =", solver.E_x)

    # Correlation Energy
    print("E_c =", solver.E_c)

    # Total Energy
    print("E =", solver.E_tot)

    return solver


if __name__ == '__main__':
    """Li atom LSD calculation example."""
    h = 0.08
    grids = np.arange(-256, 257) * h

    ks_solver = lsd_ks_dft_atom(grids, num_electrons=3, nuclear_charge=3)
    get_ks_dft_energies(ks_solver)

    # plot example self-consistent LSD density
    plt.plot(grids, ks_solver.density)
    plt.ylabel('$n(x)$', fontsize=16)
    plt.xlabel('$x$', fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()
    sys.exit()

    """Li atom LDA calculation example."""
    h = 0.08
    grids = np.arange(-256, 257) * h

    ks_solver = lda_ks_dft_atom(grids, num_electrons=3, nuclear_charge=3)
    get_ks_dft_energies(ks_solver)

    # plot example self-consistent LSD density
    plt.plot(grids, ks_solver.density)
    plt.ylabel('$n(x)$', fontsize=16)
    plt.xlabel('$x$', fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()
    sys.exit()

    """Generate atom table for various (N_e, Z) """
    # use coarser grid for faster computation.
    grids = np.linspace(-10, 10, 201)
    get_latex_table_atoms(grids)
