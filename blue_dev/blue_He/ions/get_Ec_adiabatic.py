import matplotlib.pyplot as plt
import numpy as np
import functools
import sys


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


def txt_file_to_array(file, skip=0):
    # skip = 1 means skip the first line
    # two column file to np arrays
    with open(file) as f:
        lines = f.readlines()[skip:]

        return lines


def get_v_h_n(grids, n):
    N = len(grids)
    dr = np.abs(grids[1] - grids[0])
    v_H = np.zeros(N)
    for i, r in enumerate(grids):
        for j, rp in enumerate(grids):
            if rp < r:
                v_H[i] += (rp * rp * n[j]) / r
            else:
                v_H[i] += (rp * n[j])

    v_H = dr * v_H
    return 4 * np.pi * v_H


def get_v_h_n_xc(grids, n_xc):
    N = len(grids)
    dr = np.abs(grids[1] - grids[0])
    v_H = np.zeros(N)
    for i, r in enumerate(grids):
        for j, rp in enumerate(grids):
            if rp < r:
                v_H[i] += (rp * rp * n_xc[i][j]) / r
            else:
                v_H[i] += (rp * n_xc[i][j])

    v_H = dr * v_H
    return 4 * np.pi * v_H


def get_U_xc(grids, n_HF, v_h):
    dr = np.abs(grids[1] - grids[0])
    U_xc = np.sum(grids * grids * n_HF * v_h) * dr

    return (.5) * 4 * np.pi * U_xc


def get_grids(Z):
    # different grids are used for different Z = 1,2,...,6
    if Z == 1:
        L = 10
        grid_size = 1000
        grids = np.linspace(L / grid_size - 0.0004, L, grid_size)
        return grids
    elif Z == 2:
        L = 20
        # for 1000, L = 20 using initial pt 0.018435 to minimize error of H and He
        grid_size = 1000
        grids = np.linspace(0.018435, L, grid_size)
        return grids
    else:
        if Z not in [1, 2, 3, 4, 6]:
            print('Z should be in [1, 2, 3, 4, 6]')
            return
        else:
            # density concentrated near nucleus, so smaller grid size
            L = 5
            grid_size = 1000
            grids = np.linspace(L / grid_size - 0.0004, L, grid_size)
            return grids


if __name__ == '__main__':
    fig, ax = get_plotting_params()

    lines = txt_file_to_array('exact_Ec_etc.dat', skip=1)

    Ex_exact = [float(line.split()[1]) for line in lines]
    Ec_exact = [float(line.split()[2]) for line in lines]
    Uc_exact = [float(line.split()[3]) for line in lines]
    Tc_exact = [float(line.split()[4]) for line in lines]

    Ex_exact = np.asarray(Ex_exact)
    Ec_exact = np.asarray(Ec_exact)
    Uc_exact = np.asarray(Uc_exact)
    Tc_exact = np.asarray(Tc_exact)

    Z = 1
    exact_idx = 0

    pi = np.pi

    grids = get_grids(Z)
    h = grids[1] - grids[0]

    lambda_list = np.linspace(0, 1, 11)
    n_HF = np.load("n_HF_1.npy")[0]
    # n_HF = np.load('n_HF_Z.npy')[Z - 2]
    n = n_HF

    U = get_U_xc(grids, n, get_v_h_n(grids, n))
    print('U = ', U)

    U_xc_lam = []

    for i, lam in enumerate(lambda_list):
        print("lam = ", lam)
        n2_r0 = np.load("n_r0_lambda.npy")[i]

        print("integral check: n: ",
              4 * pi * np.sum(grids * grids * n2_r0[30]) * h)

        v_h_n_ee = get_v_h_n_xc(grids, n2_r0)
        V_ee = get_U_xc(grids, n, v_h_n_ee)
        print("U_xc(lam) = ", V_ee - U)
        U_xc_lam.append(V_ee - U)

    E_xc = np.trapz(U_xc_lam, lambda_list)
    print('E_xc = ', E_xc)

    E_c = E_xc - U_xc_lam[0]
    U_c = U_xc_lam[-1] - U_xc_lam[0]
    T_c = E_c - U_c
    print('U_c = ', U_c)
    print('E_c = ', E_c)

    print('T_c = ', T_c)
    print('b = T_c/|U_c| = ', T_c / np.abs(U_c))
    print('Z = ', Z)
    Tc_error = str(
        round(100 * (Tc_exact[exact_idx] - T_c) / Tc_exact[exact_idx], 1))
    Ec_error = str(
        round(100 * (Ec_exact[exact_idx] - E_c) / Ec_exact[exact_idx], 1))

    print()
    print(round(T_c, 4), ' & ', Tc_error, ' & ', round(E_c, 4), ' & ',
          Ec_error)

    plt.plot(lambda_list, U_xc_lam)

    plt.ylabel('$U^B_{xc}(\lambda)$', fontsize=18)
    plt.xlabel('$\lambda$', fontsize=18)
    plt.grid(alpha=.4)
    plt.show()
