"""
.. _hf_scf_example:

Hartree-Fock Self-Consistent Field Example
##########################################

Summary:
    Generates Hartree-Fock (HF) a tabel from the [Baker2015]_ paper which includes Li atom and Li, Be, He, and H atoms.
"""

import sys
import os
currentpath = os.path.abspath('.')
sys.path.insert(0, os.path.dirname(currentpath))

import hf_scf, functionals, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools


def hf_scf_atom(grids, N_e, Z):
    """ Example HF-SCF calculation for a 1D atom with
        exponential interactions, see ext_potentials.exp_hydrogenic.

    Args:
        grids: grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
        N_e: the number of electrons in the atom.
        Z: the nuclear charge Z of the atom.

    Returns:
        HF-SCF solver class.
    """

    v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=Z)
    v_h = functools.partial(functionals.hartree_potential)
    fock_op = functionals.fock_operator(grids=grids)

    solver = hf_scf.HF_Solver(grids, v_ext=v_ext, v_h=v_h,
                              fock_operator=fock_op, num_electrons=N_e)
    solver.solve_self_consistent_density(sym=1)
    return solver


def get_latex_table_atoms(grids):
    """ Example Reproduce HF results in table 2 of [Baker2015]_.

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

    print("$N_e$", end=" & ")
    print("Atom/Ion", end=" & ")
    print("$T_s$", end=" & ")
    print("$V$", end=" & ")
    print("$U$", end=" & ")
    print(r"$E_x$", end=" & ")
    print(r"$E^{\text{HF}}$", end=" ")
    print(r'\\')
    print('\hline')

    for key in atom_dict.keys():
        print(atom_dict[key][0], end=" & ")
        print(key, end=" & ")

        solver = hf_scf_atom(grids, atom_dict[key][0], atom_dict[key][1])
        print(str(round(solver.T_s, 3)), end=" & ")
        print(str(round(solver.V, 3)), end=" & ")
        print(str(round(solver.U, 3)), end=" & ")
        print(str(round(solver.E_x, 3)), end=" & ")
        print(str(round(solver.E_tot, 3)), end=" ")

        print(r'\\')
        print('\hline')


def single_atom(grids, N_e, Z):
    solver = hf_scf_atom(grids, N_e, Z)

    # Non-Interacting Kinetic Energy
    print("T_s =", solver.T_s)

    # External Potential Energy
    print("V =", solver.V)

    # Hartree Energy
    print("U =", solver.U)

    # Exchange Energy
    print("E_x =", solver.E_x)

    # Total Energy
    print("E =", solver.E_tot)

    return solver


if __name__ == '__main__':
    """ Li atom HF calculation example. """
    h = 0.08
    grids = np.arange(-256, 257) * h

    example = single_atom(grids, 3, 3)

    # plot example self-consistent HF density
    plt.plot(grids, example.density)
    plt.ylabel('$n(x)$', fontsize=16)
    plt.xlabel('$x$', fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    #sys.exit()

    """ Generate atom table for various (N_e, Z) """
    # use coarser grid for faster computation.
    grids = np.linspace(-10, 10, 201)
    get_latex_table_atoms(grids)
