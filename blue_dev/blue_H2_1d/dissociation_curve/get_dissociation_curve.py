import os, sys
import single_electron
import ext_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools


def get_plotting_params():
    # TODO: match latex font
    # plotting parameters
    params = {'mathtext.default': 'default'}
    plt.rcParams.update(params)
    plt.rcParams['axes.axisbelow'] = True
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 9
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size
    fig, ax = plt.subplots()

    return fig, ax


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


def table_print(to_print, round_to_dec=3, last_in_row=False):
    rounded_to_print = format(to_print, '.' + str(round_to_dec) + 'f')
    if last_in_row:
        end = ' '
        print(rounded_to_print, end=end)
        print(r'\\')
        print('\hline')

    else:
        end = ' & '
        print(rounded_to_print, end=end)


if __name__ == '__main__':
    densities = np.load("../H2_data/densities.npy")
    potentials = np.load("../H2_data/potentials.npy")
    locations = np.load("../H2_data/locations.npy")
    nuclear_charges = np.load("../H2_data/nuclear_charges.npy")
    total_energies = np.load("../H2_data/total_energies.npy")
    Vee_energies = np.load("../H2_data/Vee_energies.npy")

    # blue conditional probability density
    n_r0_R = np.load('n_r0_R.npy')

    h = 0.08
    grids = np.arange(-256, 257) * h

    Etot_DMRG = []
    Etot_blue = []
    Etot_HF = []
    Etot_blue_Tc_DMRG = []

    U_c_blue = []
    U_c_DMRG = []
    U_c_error = []

    R_separations = []

    T_mat = get_T_mat(grids)

    for i, location in enumerate(locations):
        R = np.abs(location[0] - location[1])
        R_separations.append(R)

        #print('R = ', R)


        # DMRG
        E_DMRG = total_energies[i]
        Etot_DMRG.append(E_DMRG + get_Vpp(R))

        # HF with exact n
        n = densities[i]
        V_ext = get_v_ext(n, potentials[i])
        U = get_U(grids, n)
        E_x = -U / 2
        T_s = get_T_s(T_mat, n)
        Etot_HF.append(T_s + U + E_x + V_ext + get_Vpp(R))

        # Blue Vee (with exact n)
        Vee_blue = get_Vee_blue(grids, n_r0_R[i], n)
        Etot_blue.append(T_s + Vee_blue + V_ext + get_Vpp(R))

        # exact Tc
        Vee_DMRG = Vee_energies[i]
        T_c_DMRG = E_DMRG - T_s - Vee_DMRG - V_ext
        Etot_blue_Tc_DMRG.append(T_s + Vee_blue + V_ext + get_Vpp(R) + T_c_DMRG)

        # exact Uc
        U_c_DMRG_val = Vee_DMRG - U - E_x
        U_c_DMRG.append(U_c_DMRG_val)

        # blue U_c
        U_c_blue_val = Vee_blue - U - E_x
        U_c_blue.append(U_c_blue_val)

        U_c_error.append((U_c_DMRG[i] - U_c_blue[i]) / U_c_DMRG[i])

        # table values
        R_table = [0.0, 1.04, 1.44, 2.0, 3.04, 4.0]
        if R in R_table:
            print(R, end=" & ")
            table_print(E_x)

            table_print(Vee_blue)
            table_print(Vee_DMRG)

            U_xc_blue_val = E_x + U_c_blue_val
            U_xc_dmrg_val = E_x + U_c_DMRG_val

            table_print(U_xc_blue_val)
            table_print(U_xc_dmrg_val)

            table_print(U_c_blue_val)
            table_print(U_c_DMRG_val, last_in_row=True)

    Etot_blue = np.asarray(Etot_blue)
    R_separations = np.asarray(R_separations)
    Etot_DMRG = np.asarray(Etot_DMRG)
    Etot_HF = np.asarray(Etot_HF)
    Etot_blue_Tc_DMRG = np.asarray(Etot_blue_Tc_DMRG)

    get_plotting_params()

    # plot dissociation curve

    # total energy of H2 at infinite separation = 2E(H)
    Etot_inf_sep_H2 = 2 * (-0.669778)

    plt.plot(R_separations, Etot_blue - Etot_inf_sep_H2, label='Blue')
    plt.plot(R_separations, Etot_HF - Etot_inf_sep_H2, label='HF')
    plt.plot(R_separations, Etot_DMRG - Etot_inf_sep_H2, label='DMRG')
    plt.plot(R_separations, Etot_blue_Tc_DMRG - Etot_inf_sep_H2,
             label='Blue + $T^*_c$')

    '''
    # corrected blue curve from corrected_dissociation_curve/
    Etot_blue_corrected = np.load('Etot_blue_corrected.npy')
    plt.plot(R_separations, Etot_blue_corrected - Etot_inf_sep_H2,
             label='corrected blue')
    '''

    '''
    # plot U_c and relative error
    plt.plot(R_separations, U_c_DMRG, label='$U^*_c$')
    plt.plot(R_separations, U_c_blue, label='$U^B_c$')
    plt.plot(R_separations, U_c_error, label='$(U^*_c - U^B_c)/U^*_c$')
    '''

    plt.xlabel("R", fontsize=18)
    plt.ylabel("$E_0(R)$", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()
    plt.show()
