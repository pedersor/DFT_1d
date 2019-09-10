import HF_scf, ks_dft, functionals, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools

A = 1.071295
k = 1. / 2.385345

def diatomic_LDA(grids, N_e, d, Z, sym):
    # N_e: Number of Electrons
    # d: Nuclear Distance
    # Z: Nuclear Charge

    v_ext = functools.partial(ext_potentials.exp_H2plus, A=A, k=k, a=0, d=d, Z=Z)
    v_h = functools.partial(functionals.hartree_potential_exp, A=A, k=k, a=0)
    ex_corr = functionals.exchange_correlation_functional(grids=grids, A=A, k=k)

    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=N_e)
    solver.solve_self_consistent_density(sym)

    return solver

def diatomic_HF(grids, N_e, d, Z, sym):
    # N_e: Number of Electrons
    # d: Nuclear Distance
    # Z: Nuclear Charge

    v_ext = functools.partial(ext_potentials.exp_H2plus, A=A, k=k, a=0, d=d, Z=Z)
    v_h = functools.partial(functionals.hartree_potential_exp, A=A, k=k, a=0)
    fock_op = functionals.fock_operator(grids=grids,A=A, k=k)

    solver = HF_scf.HF_Solver(grids, v_ext=v_ext, v_h=v_h, fock_operator=fock_op, num_electrons=N_e)
    solver.solve_self_consistent_density(sym)

    return solver


def get_energies(grids, d):

    HF_Energies = []
    LDA_Energies = []
    P_HF_Energies = []
    P_LDA_Energies = []

    N_e = 2
    Z = 1
    
    if N_e == 1:

        sym = 1

        for i in range(len(d)):
            HF_Solver = diatomic_HF(grids, N_e, d[i], Z, sym)
            LDA_Solver = diatomic_LDA(grids, N_e, d[i], Z, sym)

            repulsion = -ext_potentials.exp_hydrogenic(d[i], A, k, 0, Z)

            HF_Energies.append(HF_Solver.E_tot + repulsion)
            LDA_Energies.append(LDA_Solver.E_tot + repulsion)

            print(d[i], HF_Energies[i], LDA_Energies[i])

        return HF_Energies, LDA_Energies

    elif N_e == 2:

        sym = 1

        for i in range(len(d)):
            HF_Solver = diatomic_HF(grids, N_e, d[i], Z, sym)
            LDA_Solver = diatomic_LDA(grids, N_e, d[i], Z, sym)

            repulsion = -ext_potentials.exp_hydrogenic(d[i], A, k, 0, Z)

            HF_Energies.append(HF_Solver.E_tot + repulsion)
            LDA_Energies.append(LDA_Solver.E_tot + repulsion)

            print(d[i], HF_Energies[i], LDA_Energies[i])

        sym = 2
        sym2 = 10

        for i in range(len(d)):
            HF_Solver = diatomic_HF(grids, N_e, d[i], Z, sym)
            LDA_Solver = diatomic_LDA(grids, N_e, d[i], Z, sym2)

            repulsion = -ext_potentials.exp_hydrogenic(d[i], A, k, 0, Z)

            P_HF_Energies.append(HF_Solver.E_tot + repulsion)
            P_LDA_Energies.append(LDA_Solver.E_tot + repulsion)

            print(d[i], P_HF_Energies[i], P_LDA_Energies[i])

        return HF_Energies, LDA_Energies, P_HF_Energies, P_LDA_Energies


if __name__ == '__main__':
    grids = np.linspace(-10, 10, 200)
    d = np.linspace(0, 6, 50) # Nuclear Distances (Diatomic)

    #HF_Energies, LDA_Energies, P_HF_Energies, P_LDA_Energies = get_energies(grids, d)
    HF_Energies, LDA_Energies, P_HF_Energies, P_LDA_Energies  = get_energies(grids, d)

    # Molecular Dissociation Curves
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

    '''
    # R0
    smallest_index = energies.index(min(energies))
    print('The minimum energy is', min(energies), 'at a distance of', d[smallest_index])
    '''

    '''
    # Plot ongoing nUP / nDOWN / density
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('x')
    ax1.set_ylabel('wf')

    for i in range(0, len(wave_functionUP_list), 1):
        plt.plot(grids, wave_functionUP_list[i][0], label= 'wfcUP ' + str(i))

    #for i in range(0, len(wave_functionDOWN_list), 1):
     #   plt.plot(grids, wave_functionDOWN_list[i][0], label= 'wfDOWN ' + str(i))

    plt.legend(loc='best')
    plt.grid()
    '''
    
    '''
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('x')
    ax1.set_ylabel('n')
    plt.plot(grids, wave_functionUP_list[len(wave_functionUP_list) - 1][0] + 1, 'b', label='wfUP') 
    plt.plot(grids, wave_functionDOWN_list[len(wave_functionDOWN_list) - 1][0] , 'g', label='wfDOWN') 
    plt.legend(loc='best')
    plt.grid()
    '''

    plt.show()