import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import sys


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


def get_v_h_n_trapz(grids, n):
    N = len(grids)
    v_H = np.zeros(N)
    for i, r in enumerate(grids):
        # radial integral
        radial_int = []
        for j, rp in enumerate(grids):
            if rp < r:
                radial_int.append((rp * rp * n[j]) / r)
            else:
                radial_int.append(rp * n[j])
        radial_int = np.asarray(radial_int)
        v_H[i] = np.trapz(radial_int, grids)

    return 4 * np.pi * v_H


def get_U_xc_trapz(grids, n, v_h):
    U_xc = np.trapz(grids * grids * n * v_h, grids)

    return (.5) * 4 * np.pi * U_xc


def get_v_h_n_xc_trapz(grids, n_xc):
    N = len(grids)
    v_H = np.zeros(N)
    for i, r in enumerate(grids):
        # radial integral
        radial_int = []
        for j, rp in enumerate(grids):
            if rp < r:
                radial_int.append((rp * rp * n_xc[i][j]) / r)
            else:
                radial_int.append(rp * n_xc[i][j])
        radial_int = np.asarray(radial_int)
        v_H[i] = np.trapz(radial_int, grids)

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


def txt_file_to_array(file, skip=0):
    # skip = 1 means skip the first line
    # two column file to np arrays
    with open(file) as f:
        lines = f.readlines()[skip:]

        r = []
        n = []
        for line in lines:
            r_val = line.split()[0]
            r.append(float(r_val.replace('D', 'E')))

            n_val = line.split()[1]
            n.append(float(n_val.replace('D', 'E')))

    r = np.asarray(r)
    n = np.asarray(n)
    return r, n

if __name__ == '__main__':
    r, n = txt_file_to_array('Cyrus_Umrigar_data.txt', skip=12)
    interp_int_n = InterpolatedUnivariateSpline(r, n, k=3)

    L = 10
    grid_size = 1000
    grids = np.linspace(L / grid_size - 0.0004, L, grid_size)

    n2_r0 = np.load('n_r0_1.npy')[0]
    n_HF = np.load('n_HF_1.npy')[0]



    # v_h = get_v_h_n(grids, interp_int_n(grids))
    v_h = get_v_h_n_trapz(r, interp_int_n(r))
    # U = get_U_xc(grids, interp_int_n(grids), v_h)
    U = get_U_xc_trapz(r, interp_int_n(r), v_h)
    print('Ex = ', -U / 2)

    v_h_n_ee = get_v_h_n_xc_trapz(grids, n2_r0)

    interp_v_h_n_ee = InterpolatedUnivariateSpline(grids, v_h_n_ee, k=3)

    V_ee = get_U_xc_trapz(r, interp_int_n(r), interp_v_h_n_ee(r))
    print("V^B_ee = ", V_ee)
    Ex = -0.3809
    Ex = -U / 2
    print("U^B_c = ", V_ee + Ex)

    T_s = 0.4999
    Vext = -1.366524
    Ec_w_exact_Ex = .5 * (-2 * T_s - Vext - 2 * (-2 * Ex) - 2 * Ex + V_ee)
    print('Ec_w_exact_Ex = ', Ec_w_exact_Ex)

    sys.exit()
    # compare to HF
    n_HF = np.load('n_HF_1.npy')[0]

    L = 10
    grid_size = 1000
    grids = np.linspace(L / grid_size - 0.0004, L, grid_size)

    plt.plot(grids, 4 * np.pi * grids * grids * n_HF, label='n_HF')
    plt.plot(r, 4 * np.pi * r * r * n, label='cyrus n')

    plt.legend()
    plt.show()
