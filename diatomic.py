import HF_scf, ks_dft, functionals, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools

A = 1.071295
k = 1. / 2.385345

def diatomic_lda_run(grids, N_e, d, Z, sym):
    # N_e: Number of Electrons
    # d: Nuclear Distance
    # Z: Nuclear Charge

    v_ext = functools.partial(ext_potentials.exp_H2plus, A=A, k=k, a=0, d=d, Z=Z)
    v_h = functools.partial(functionals.hartree_potential_exp, A=A, k=k, a=0)
    ex_corr = functionals.exchange_correlation_functional(grids=grids, A=A, k=k)

    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=N_e)
    solver.solve_self_consistent_density(sym)

    return solver

def diatomic_hf_run(grids, N_e, d, Z, sym):
    # N_e: Number of Electrons
    # d: Nuclear Distance
    # Z: Nuclear Charge

    v_ext = functools.partial(ext_potentials.exp_H2plus, A=A, k=k, a=0, d=d, Z=Z)
    v_h = functools.partial(functionals.hartree_potential_exp, A=A, k=k, a=0)
    fock_op = functionals.fock_operator(grids=grids,A=A, k=k)

    solver = HF_scf.HF_Solver(grids, v_ext=v_ext, v_h=v_h, fock_operator=fock_op, num_electrons=N_e)
    solver.solve_self_consistent_density(sym)

    return solver


def get_energies(grids, d, N_e, Z):
    HF_Energies = []
    LDA_Energies = []
    P_HF_Energies = []
    P_LDA_Energies = []
    
    if N_e == 1:

        sym = 1
        for i in range(len(d)):
            HF_Solver = diatomic_hf_run(grids, N_e, d[i], Z, sym)
            LDA_Solver = diatomic_lda_run(grids, N_e, d[i], Z, sym)
            repulsion = -ext_potentials.exp_hydrogenic(d[i], A, k, 0, Z)

            HF_Energies.append(HF_Solver.E_tot + repulsion)
            LDA_Energies.append(LDA_Solver.E_tot + repulsion)

            print(d[i], HF_Energies[i], LDA_Energies[i])

        return HF_Energies, LDA_Energies

    elif N_e == 2:

        sym = 1
        for i in range(len(d)):
            HF_Solver = diatomic_hf_run(grids, N_e, d[i], Z, sym)
            LDA_Solver = diatomic_lda_run(grids, N_e, d[i], Z, sym)
            repulsion = -ext_potentials.exp_hydrogenic(d[i], A, k, 0, Z)

            HF_Energies.append(HF_Solver.E_tot + repulsion)
            LDA_Energies.append(LDA_Solver.E_tot + repulsion)

            print(d[i], HF_Energies[i], LDA_Energies[i])

        sym = 2
        sym2 = 100
        for i in range(len(d)):
            HF_Solver = diatomic_hf_run(grids, N_e, d[i], Z, sym)
            LDA_Solver = diatomic_lda_run(grids, N_e, d[i], Z, sym2)
            repulsion = -ext_potentials.exp_hydrogenic(d[i], A, k, 0, Z)

            P_HF_Energies.append(HF_Solver.E_tot + repulsion)
            P_LDA_Energies.append(LDA_Solver.E_tot + repulsion)

            print(d[i], P_HF_Energies[i], P_LDA_Energies[i])

        return HF_Energies, LDA_Energies, P_HF_Energies, P_LDA_Energies


if __name__ == '__main__':
    grids = np.linspace(-10, 10, 201)
    d = np.linspace(0, 6, 40) # Nuclear Distances

    N_e = 2
    Z = 1

    # Molecular Dissociation Curves
    if N_e == 1:
        HF_Energies, LDA_Energies = get_energies(grids, d, N_e, Z)
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('R')
        ax1.set_ylabel('E0(R)')
        plt.plot(d, LDA_Energies, 'r', label='LDA')
        plt.plot(d, HF_Energies, 'b', label='HF')
        plt.legend(loc='best')
        plt.grid()
    elif N_e == 2:
        HF_Energies, LDA_Energies, P_HF_Energies, P_LDA_Energies = get_energies(grids, d, N_e, Z)
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('R')
        ax1.set_ylabel('E0(R)')
        plt.plot(d, LDA_Energies, 'r', label='LDA')
        plt.plot(d, P_LDA_Energies, '--r')
        plt.plot(d, HF_Energies, 'b', label='HF')
        plt.plot(d, P_HF_Energies, '--b')
        plt.legend(loc='best')
        plt.grid()

    plt.show()