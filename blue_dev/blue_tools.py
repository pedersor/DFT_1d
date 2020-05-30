import numpy as np


# U_xc integrals etc.
def get_v_H_n(grids, n):
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


def get_v_H_n_xc(grids, n_xc):
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


def get_U_xc(grids, n, v_h):
    # this can also be used to obtain U_c and U_H

    U_xc = np.trapz(grids * grids * n * v_h, grids)

    return (.5) * 4 * np.pi * U_xc


# tools for exact pair density
def txt_file_to_2d_array(file, grids):
    array_2d = []
    with open(file) as f:
        lines = f.readlines()

        counter = 0
        array_1d = []
        for line in lines:
            array_1d.append(float(line.split()[2]))

            counter += 1
            if counter == len(grids):
                array_1d = np.asarray(array_1d)
                array_2d.append(array_1d)
                counter = 0
                array_1d = []

    array_2d = np.asarray(array_2d)
    return array_2d


def get_P_r_rp_idx(P_r_rp, n, x_idx, h):
    P_r_rp_idx = P_r_rp[x_idx]

    P_r_rp_idx[x_idx] = P_r_rp_idx[x_idx] - n[x_idx] * h

    P_r_rp_idx = P_r_rp_idx / (h * h)
    return P_r_rp_idx

# easy table print
def table_print(to_print, round_to_dec=3, last_in_row=False):
    if isinstance(to_print, float):
        rounded_to_print = format(to_print, '.' + str(round_to_dec) + 'f')
    else:
        rounded_to_print = to_print

    if last_in_row:
        end = ' '
        print(rounded_to_print, end=end)
        print(r'\\')
        print('\hline')

    else:
        end = ' & '
        print(rounded_to_print, end=end)
