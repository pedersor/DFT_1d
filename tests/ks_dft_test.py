import ks_dft, dft_potentials, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools

def lda_dft_run(grids, N_e, Z):
    # N_e: Number of Electrons
    # Z: Nuclear Charge

    A = 1.071295
    k = 1. / 2.385345

    v_ext = functools.partial(ext_potentials.exp_hydrogenic, A=A, k=k, a=0, Z=Z)
    v_h = functools.partial(dft_potentials.hartree_potential_exp, A=A, k=k, a=0)
    ex_corr = dft_potentials.exchange_correlation_functional(grids=grids, A=A, k=k)

    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=N_e)
    solver.solve_self_consistent_density()

    return solver


def get_latex_table(grids):
    # "atom/ion": [N_e, Z]
    atom_dict = {"H": [1, 1], "He$^+$": [1, 2], "Li$^{2+}$": [1, 3], "Be$^{3+}$": [1, 4], "He": [2, 2], "Li$^+$": [2, 3],
                 "Be$^{2+}$": [2, 4], "Li": [3, 3], "Be$^+$": [3, 4]}
    atom_dict = {"Li": [3, 3]}

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

        solver = lda_dft_run(grids, atom_dict[key][0], atom_dict[key][1])
        print(str(round(solver.T_s, 3)), end=" & ")
        print(str(round(solver.V, 3)), end=" & ")
        print(str(round(solver.U, 3)), end=" & ")
        print(str(round(solver.E_x, 3)), end=" & ")
        print(str(round(solver.E_c, 3)), end=" & ")
        print(str(round(solver.E_tot, 3)), end=" ")

        print(r'\\')
        print('\hline')


def diatomic_lda_dft(grids, N_e, d, Z):
    # N_e: Number of Electrons
    # d: Nuclear Distance
    # Z: Nuclear Charge

    A = 1.071295
    k = 1. / 2.385345

    v_ext = functools.partial(ext_potentials.exp_H2plus, A=A, k=k, a=0, d=d, Z=Z)
    v_h = functools.partial(dft_potentials.hartree_potential_exp, A=A, k=k, a=0)
    ex_corr = dft_potentials.exchange_correlation_functional(grids=grids, A=A, k=k)

    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=N_e)
    solver.solve_self_consistent_density()

    return solver


def get_energies(grids):
    d = np.linspace(0.1, 6, 10)
    N_e = 1
    Z = 1
    energies = []
    
    i = 0
    while (i < len(d)):
        solver = diatomic_lda_dft(grids, N_e, d[i], Z)
        energies.append(solver.E_tot)
        print(d[i], energies[i])
        i += 1

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_title('H2+ Dissociation Curve')
    ax1.set_xlabel('R')
    ax1.set_ylabel('E0(R)')
    plt.plot(d, energies, 'r')
    plt.grid()
    plt.show()


def simple_dft_lda_test(grids, N_e, Z):
    solver = lda_dft_run(grids, N_e, Z)

    # Non-Interacting Kinetic Energy
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


if __name__ == '__main__':
    grids = np.linspace(-15, 15, 1001)
    get_latex_table(grids)
    #energies = get_energies(grids)