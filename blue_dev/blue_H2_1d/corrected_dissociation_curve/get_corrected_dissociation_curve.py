import os, sys

# linux cluster path
sys.path.append('/DFS-B/DATA/burke/pedersor/Kohn_Sham_DFT_1d')
sys.path.append('/DFS-B/DATA/burke/pedersor/Kohn_Sham_DFT_1d/blue_dev')
sys.path.append(
    '/DFS-B/DATA/burke/pedersor/Kohn_Sham_DFT_1d/blue_dev/blue_H2_1d/corrected_dissociation_curve/')

import single_electron
import ext_potentials

import matplotlib

# linux cluster does not have display:
matplotlib.use('Pdf')
import matplotlib.pyplot as plt

import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import get_E_xc_lambda
import get_n_r0_R_lambda_HF


def get_plotting_params():
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
    # TODO: add general case for N electrons into external_potentials.py (done somewhere in DMRG_1D lib)
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
    densities = np.load("H2_data/densities.npy")
    potentials = np.load("H2_data/potentials.npy")
    locations = np.load("H2_data/locations.npy")
    nuclear_charges = np.load("H2_data/nuclear_charges.npy")
    total_energies = np.load("H2_data/total_energies.npy")
    Vee_energies = np.load("H2_data/Vee_energies.npy")

    # blue conditional probability density
    n_r0_R = np.load('n_r0_R.npy')

    h = 0.08
    grids = np.arange(-256, 257) * h

    Etot_DMRG = []
    Etot_blue = []
    Etot_HF = []
    Etot_blue_Tc_DMRG = []
    Etot_blue_corrected = []

    U_c_blue = []
    U_c_DMRG = []
    U_c_blue_HF = []

    E_c_DMRG = []
    E_c_blue_HF = []

    T_c_DMRG = []
    T_c_blue_HF = []

    R_separations = []

    T_mat = get_T_mat(grids)

    for i, location in enumerate(locations):
        print('iter: ', i)

        R = np.abs(location[0] - location[1])
        R_separations.append(R)

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

        # Blue Vee (HF n + adiabatic connection) corrected
        n_HF = get_n_r0_R_lambda_HF.get_n_HF(grids, potentials[i])
        n_r0_lambda_HF = np.load('R' + str(i + 1) + '/n_r0_lambda_HF.npy')
        lambda_list = np.linspace(0, 1, 11)

        U_c_HF, T_c_HF, E_c_HF, E_xc_HF = get_E_xc_lambda.get_Exc(grids, n_HF,
                                                                  n_r0_lambda_HF,
                                                                  lambda_list)
        T_s_HF = get_T_s(T_mat, n_HF)
        U_HF = get_U(grids, n_HF)
        E_x_HF = -U_HF / 2
        V_ext_HF = get_v_ext(n_HF, potentials[i])

        Etot_blue_corrected.append(
            T_s_HF + U_HF + E_x_HF + V_ext_HF + E_c_HF + get_Vpp(R))
        U_c_blue_HF.append(U_c_HF)
        T_c_blue_HF.append(T_c_HF)
        E_c_blue_HF.append(E_c_HF)

        # exact Tc
        Vee_DMRG = Vee_energies[i]
        T_c_DMRG.append(E_DMRG - T_s - Vee_DMRG - V_ext)
        Etot_blue_Tc_DMRG.append(
            T_s + Vee_blue + V_ext + get_Vpp(R) + T_c_DMRG[i])

        # exact Uc
        U_c_DMRG.append(Vee_DMRG - U - E_x)

        # exact Ec
        E_c_DMRG.append(T_c_DMRG[i]+U_c_DMRG[i])


        # blue U_c
        U_c_blue.append(Vee_blue - U - E_x)

        # equilibrium values
        R_eq = 4.00
        if R == R_eq:
            print('R = ', R_eq, ' values: ')
            print("U_c_DMRG = ", U_c_DMRG[i])
            print("U_c_blue = ", U_c_blue[i])

            print("U_c_error = ", (U_c_DMRG[i] - U_c_blue[i]) / U_c_DMRG[i])

            print('T_c_DMRG = ', T_c_DMRG)
            print('E_c_DMRG = ', T_c_DMRG + U_c_DMRG[i])

    Etot_blue = np.asarray(Etot_blue)
    Etot_DMRG = np.asarray(Etot_DMRG)
    Etot_blue_corrected = np.asarray(Etot_blue_corrected)

    U_c_DMRG = np.asarray(U_c_DMRG)
    U_c_blue = np.asarray(U_c_blue)
    U_c_blue_HF = np.asarray(U_c_blue_HF)

    E_c_DMRG = np.asarray(E_c_DMRG)
    E_c_blue_HF = np.asarray(E_c_blue_HF)

    T_c_DMRG = np.asarray(T_c_DMRG)
    T_c_blue_HF = np.asarray(T_c_blue_HF)

    R_separations = np.asarray(R_separations)

    # save Etot_blue_corrected open in get_dissociation_curve
    # np.save('Etot_blue_corrected', Etot_blue_corrected)

    get_plotting_params()

    '''
    # plot dissociation curve
    plt.plot(R_separations, Etot_blue, label='Blue')
    plt.plot(R_separations, Etot_HF, label='HF')
    plt.plot(R_separations, Etot_DMRG, label='DMRG')
    plt.plot(R_separations, Etot_blue_Tc_DMRG, label='Blue + exact Tc')
    plt.plot(R_separations, Etot_blue_corrected, label='corrected blue')
    '''

    # plot U_c and relative error
    plt.plot(R_separations, U_c_DMRG, label='$U^*_c$')
    plt.plot(R_separations, U_c_blue, label='$U^B_c[n^{DMRG}]$')
    plt.plot(R_separations, U_c_blue_HF, label='$U^B_c[n^{HF}]$')
    U_c_error = (U_c_DMRG - U_c_blue_HF) / U_c_DMRG
    plt.plot(R_separations, U_c_error, label='$(U^*_c - U^B_c[n^{HF}])/U^*_c$')

    plt.xlabel("R", fontsize=18)
    plt.ylabel("$E_0(R)$", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()
    plt.savefig('U_c_blue_corrected.pdf')
    plt.close()

    # plot T_c and relative error
    plt.plot(R_separations, T_c_DMRG, label='$T^*_c$')
    plt.plot(R_separations, T_c_blue_HF, label='$T^B_c[n^{HF}]$')
    T_c_error = (T_c_DMRG - T_c_blue_HF) / T_c_DMRG
    plt.plot(R_separations, T_c_error, label='$(T^*_c - T^B_c[n^{HF}])/T^*_c$')

    plt.xlabel("R", fontsize=18)
    plt.ylabel("$E_0(R)$", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()
    plt.savefig('T_c_blue_corrected.pdf')
    plt.close()

    # plot E_c and relative error
    plt.plot(R_separations, E_c_DMRG, label='$E^*_c$')
    plt.plot(R_separations, E_c_blue_HF, label='$E^B_c[n^{HF}]$')
    E_c_error = (E_c_DMRG - E_c_blue_HF) / E_c_DMRG
    plt.plot(R_separations, E_c_error, label='$(E^*_c - E^B_c[n^{HF}])/E^*_c$')

    plt.xlabel("R", fontsize=18)
    plt.ylabel("$E_0(R)$", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()
    plt.savefig('E_c_blue_corrected.pdf')
    plt.close()
