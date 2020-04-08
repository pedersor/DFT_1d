# Test of HF implementation on single atoms/ions

import hf_scf, functionals, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools


def hf_run(grids, N_e, Z):
    # N_e: Number of Electrons
    # Z: Nuclear Charge

    A = 1.071295
    k = 1. / 2.385345

    v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=Z)
    v_h = functools.partial(functionals.hartree_potential)
    fock_op = functionals.fock_operator(grids=grids, A=A, k=k)

    solver = hf_scf.HF_Solver(grids, v_ext=v_ext, v_h=v_h,
                              fock_operator=fock_op, num_electrons=N_e)
    solver.solve_self_consistent_density(sym=1)
    return solver


def get_latex_table(grids):
    # "atom/ion": [N_e, Z]
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

        solver = hf_run(grids, atom_dict[key][0], atom_dict[key][1])
        print(str(round(solver.T_s, 3)), end=" & ")
        print(str(round(solver.V, 3)), end=" & ")
        print(str(round(solver.U, 3)), end=" & ")
        print(str(round(solver.E_x, 3)), end=" & ")
        print(str(round(solver.E_tot, 3)), end=" ")

        print(r'\\')
        print('\hline')


def single_atom(grids, N_e, Z):
    solver = hf_run(grids, N_e, Z)

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


if __name__ == '__main__':
    grids = np.linspace(-10, 10, 201)

    single_atom(grids, 3, 3)
    # get_latex_table(grids)
