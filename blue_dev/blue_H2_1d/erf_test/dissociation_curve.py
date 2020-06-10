import os, sys
import single_electron
import ext_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import blue_tools


def get_v_H(grids, n):
    N = len(grids)
    dx = np.abs(grids[1] - grids[0])
    v_H = np.zeros(N)
    for i in range(N):
        for j in range(N):
            v_H[i] += n[j] * (-1) * ext_potentials.exp_hydrogenic(
                grids[i] - grids[j])
    v_H *= dx
    return v_H


def get_U(grids, n):
    U = 0.5 * np.sum(n * get_v_H(grids, n)) * h
    return U


def get_T_mat(grids):
    solver = single_electron.EigenSolver(grids,
                                         potential_fn=functools.partial(
                                             ext_potentials.exp_hydrogenic),
                                         boundary_condition='open',
                                         num_electrons=1)

    T_mat = solver.get_kinetic_matrix()
    return T_mat


def get_T_s(T_mat, n):
    psi = (n / 2) ** (0.5)
    return 2 * np.sum(psi * np.matmul(T_mat, psi)) * h


def get_Vpp(R):
    return -1 * ext_potentials.exp_hydrogenic(R)


def get_blue_potential(grids, pot, r0):
    return pot - ext_potentials.exp_hydrogenic(grids - r0)


def get_v_ext(n, ext_potential):
    return np.sum(n * ext_potential) * h


def get_Vee_blue(grids, n_r0_R, n):
    N = len(grids)
    dx = np.abs(grids[1] - grids[0])
    v_Vee = np.zeros(N)
    for i in range(N):
        for j in range(N):
            v_Vee[i] += n_r0_R[i][j] * (-1) * ext_potentials.exp_hydrogenic(
                grids[i] - grids[j])
    v_Vee *= dx

    Vee_blue = 0.5 * np.sum(n * v_Vee) * h

    return Vee_blue


if __name__ == '__main__':
    densities = np.load("../H2_data/densities.npy")
    potentials = np.load("../H2_data/potentials.npy")
    locations = np.load("../H2_data/locations.npy")
    nuclear_charges = np.load("../H2_data/nuclear_charges.npy")
    total_energies = np.load("../H2_data/total_energies.npy")
    Vee_energies = np.load("../H2_data/Vee_energies.npy")

    h = 0.08
    grids = np.arange(-256, 257) * h

    Etot_dmrg = []
    Etot_HF = []
    U_c_dmrg = []

    # blue conditional probability density
    blue_files = ['3o2r_s', '1or_s']
    labels = ['$3/(2r_s)$', '$1/r_s$']
    blue_approxs = []
    for i, file in enumerate(blue_files):
        n_r0_blue = np.load('n_r0_1D_H2_erf_gam_' + file + '.npy')
        blue_quantities = {'approx': file, 'label': labels[i],
                           'n_r0_blue': n_r0_blue,
                           'Etot_blue': [],
                           'U_c_blue': [],
                           'U_c_error': []}
        blue_approxs.append(blue_quantities)

    R_separations = []

    T_mat = get_T_mat(grids)

    for i, location in enumerate(locations):
        R = np.abs(location[0] - location[1])
        R_separations.append(R)

        # print('R = ', R)

        # dmrg
        E_dmrg = total_energies[i]
        Etot_dmrg.append(E_dmrg + get_Vpp(R))

        # exact quantities
        n = densities[i]
        V_ext = get_v_ext(n, potentials[i])
        U = get_U(grids, n)
        E_x = -U / 2
        T_s = get_T_s(T_mat, n)

        # exact Tc
        Vee_dmrg = Vee_energies[i]
        T_c_dmrg = E_dmrg - T_s - Vee_dmrg - V_ext

        # exact Uc
        U_c_dmrg_R = Vee_dmrg - U - E_x
        U_c_dmrg.append(U_c_dmrg_R)

        # Blue Vee (with exact n)
        for blue_approx in blue_approxs:
            n_r0_blue_R = blue_approx['n_r0_blue'][i]
            Vee_blue_R = get_Vee_blue(grids, n_r0_blue_R, n)

            Etot_blue = blue_approx['Etot_blue']
            Etot_blue.append(T_s + Vee_blue_R + V_ext + get_Vpp(R) + T_c_dmrg)
            blue_approx['Etot_blue'] = Etot_blue

            # blue U_c
            U_c_blue_R = Vee_blue_R - U - E_x
            U_c_blue = blue_approx['U_c_blue']
            U_c_blue.append(U_c_blue_R)
            blue_approx['U_c_blue'] = U_c_blue

            # U_c error
            U_c_error_R = 100 * (U_c_dmrg_R - U_c_blue_R) / U_c_dmrg_R
            U_c_error = blue_approx['U_c_error']
            U_c_error.append(U_c_error_R)
            blue_approx['U_c_error'] = U_c_error

        # table R values
        R_table = [0.0, 1.04, 1.44, 2.0, 3.04, 4.0]
        if R in R_table:
            blue_tools.table_print(R, round_to_dec=2)

            blue_tools.table_print(Vee_blue_R)
            blue_tools.table_print(Vee_dmrg)

            U_xc_blue_R = E_x + U_c_blue_R
            U_xc_dmrg_R = E_x + U_c_dmrg_R

            blue_tools.table_print(U_xc_blue_R)
            blue_tools.table_print(U_xc_dmrg_R)

            blue_tools.table_print(U_c_blue_R)
            blue_tools.table_print(U_c_dmrg_R)

            blue_tools.table_print(U_c_error_R, round_to_dec=1,
                                   last_in_row=True)

    R_separations = np.asarray(R_separations)
    Etot_dmrg = np.asarray(Etot_dmrg)
    Etot_HF = np.asarray(Etot_HF)

    # plot dissociation curve

    # total energy of H2 at infinite separation = 2E(H)
    Etot_inf_sep_H2 = 2 * (-0.669778)

    # TODO: labels
    for blue_approx in blue_approxs:
        Etot_blue = np.asarray(blue_approx['Etot_blue'])
        plt.plot(R_separations, Etot_blue - Etot_inf_sep_H2,
                 label='blue, ' + blue_approx['label'])

    plt.plot(R_separations, Etot_dmrg - Etot_inf_sep_H2, label='dmrg')

    plt.xlabel("R", fontsize=18)
    plt.ylabel("$E_0(R)$", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()
    plt.show()

    # plot U_c and/or relative error
    for blue_approx in blue_approxs:
        U_c_blue = np.asarray(blue_approx['U_c_blue'])
        plt.plot(R_separations, U_c_blue, label='blue, ' + blue_approx['label'])

    plt.plot(R_separations, U_c_dmrg, label='dmrg')

    plt.xlabel("R", fontsize=18)
    plt.ylabel("$E_0(R)$", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()
    plt.show()
