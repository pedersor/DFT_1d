import ks_dft, functionals, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools

A = 1.071295
k = 1. / 2.385345

def lda_dft_run(grids, N_e, Z):
    # N_e: Number of Electrons
    # Z: Nuclear Charge

    v_ext = functools.partial(ext_potentials.exp_hydrogenic, A=A, k=k, a=0, Z=Z)
    v_h = functools.partial(functionals.hartree_potential_exp, A=A, k=k, a=0)
    ex_corr = functionals.exchange_correlation_functional(grids=grids, A=A, k=k)

    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=N_e)
    solver.solve_self_consistent_density()

    return solver


def get_latex_table(grids):
    # "atom/ion": [N_e, Z]
    atom_dict = {"H": [1, 1], "He$^+$": [1, 2], "Li$^{2+}$": [1, 3], "Be$^{3+}$": [1, 4], "He": [2, 2], "Li$^+$": [2, 3],
                 "Be$^{2+}$": [2, 4], "Li": [3, 3], "Be$^+$": [3, 4], "Be": [4, 4]}

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


def diatomic_lda_dft_run(grids, N_e, d, Z, sym):
    # N_e: Number of Electrons
    # d: Nuclear Distance
    # Z: Nuclear Charge

    v_ext = functools.partial(ext_potentials.exp_H2plus, A=A, k=k, a=0, d=d, Z=Z)
    v_h = functools.partial(functionals.hartree_potential_exp, A=A, k=k, a=0)
    ex_corr = functionals.exchange_correlation_functional(grids=grids, A=A, k=k)

    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=N_e)
    solver.solve_self_consistent_density(sym)

    return solver


def get_energies(grids, d):
    N_e = 2
    Z = 1
    
    sym = 1
    energies1 = []
    for i in range(len(d)):
        solver = diatomic_lda_dft_run(grids, N_e, d[i], Z, sym)
        repulsion = -ext_potentials.exp_hydrogenic(d[i], A, k, 0, Z)
        energies1.append(solver.E_tot + repulsion)
        print(d[i], energies1[i])

    sym = 10
    energies2 = []
    for i in range(len(d)):
        solver = diatomic_lda_dft_run(grids, N_e, d[i], Z, sym)
        repulsion = -ext_potentials.exp_hydrogenic(d[i], A, k, 0, Z)
        energies2.append(solver.E_tot + repulsion)
        print(d[i], energies2[i])

    return energies1, energies2 # solver.density_list, solver.nUP_list, solver.nDOWN_list


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
    grids = np.linspace(-10, 10, 200)
    d = np.linspace(0.1, 6, 35) # Nuclear Distances (Diatomic)

    #get_latex_table(grids)
    energies1, energies2 = get_energies(grids, d) # density_list, nUP_list, nDOWN_list = get_energies(grids, d)

    # Molecular Dissociation Curves
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('R')
    ax1.set_ylabel('E0(R)')
    plt.plot(d, energies1, 'r', label='Unperturbed')
    plt.plot(d, energies2, 'r--', label='Perturbed')
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
    ax1.set_ylabel('n')

    #for i in range(0, len(density_list), 1):
     #   plt.plot(grids, density_list[i], label= 'Density ' + str(i))

    for i in range(0, len(nUP_list), 1):
        plt.plot(grids, nUP_list[i], label= 'nUP ' + str(i))

    #for i in range(0, len(nDOWN_list), 1):
     #   plt.plot(grids, nDOWN_list[i], label= 'nDOWN ' + str(i))

    plt.legend(loc='best')
    plt.grid()
    '''

    '''
    # Plot final nUP / nDOWN / density
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('x')
    ax1.set_ylabel('n')
    plt.plot(grids, density_list[len(density_list) - 1], 'r', label='Final Density')
    plt.plot(grids, nUP_list[len(nUP_list) - 1], 'b', label='Final nUP') 
    plt.plot(grids, nDOWN_list[len(nDOWN_list) - 1] , 'g', label='Final nDOWN') 
    #plt.plot(grids, nUP_list[len(nUP_list) - 1] - nDOWN_list[len(nDOWN_list) - 1] , 'g', label='Difference') 
    plt.legend(loc='best')
    plt.grid()
    '''

    plt.show()