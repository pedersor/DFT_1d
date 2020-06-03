import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


# get n_x and n_xc
def get_two_el_n_x(n):
    # for 2 electrons, n_x = -n/2
    n_x = []
    for i in range(len(n)):
        n_x.append(-n / 2.)
    n_x = np.asarray(n_x)
    return n_x


def pair_density_to_n_xc(pair_density, n):
    # Pair density, P(r,r'), to n_xc(r,r')
    n_xc = []
    for i, P_rp in enumerate(pair_density):
        n_xc.append(-n + (P_rp / n[i]))
    n_xc = np.asarray(n_xc)
    # division by zero: nan -> 0.0
    n_xc = np.nan_to_num(n_xc)
    return n_xc


def n_CP_to_n_xc(n_CP, n):
    # conditional probability (CP). n^{CP}(r,r') = n(r') + n_{xc}(r,r')
    n_xc = []
    for i, CP_rp in enumerate(n_CP):
        n_xc.append(-n + CP_rp)
    n_xc = np.asarray(n_xc)
    return n_xc


# radial U_xc integrals etc.
def radial_get_v_H_n(grids, n):
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


def radial_get_v_H_n_xc(grids, n_xc):
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


def radial_get_U_xc(grids, n, v_h):
    # this can also be used to obtain U_c and U_H

    U_xc = np.trapz(grids * grids * n * v_h, grids)

    return (.5) * 4 * np.pi * U_xc


# averaged xc hole
def get_interp_n_xc(grids, n_xc):
    interp_n_xc = []
    for i, n_xc_r in enumerate(n_xc):
        # interpolated cubic spline function
        cubic_spline = InterpolatedUnivariateSpline(grids, n_xc_r, k=3)
        interp_n_xc.append(cubic_spline)

    interp_n_xc = np.asarray(interp_n_xc)
    return interp_n_xc


def get_avg_xc_hole(u, grids, interp_n_xc, n):
    # x = cos(theta), see RP logbook 3/16/20
    x_grids = np.linspace(-1, 1, len(grids))
    n_xc_after_x_integral_r = []
    for i, r in enumerate(grids):
        interp_n_xc_r = interp_n_xc[i]
        u_plus_r = ((2 * u * r * x_grids) + (u * u) + (r * r)) ** (0.5)
        n_xc_after_x_integral = np.trapz(interp_n_xc_r(u_plus_r), x_grids)
        n_xc_after_x_integral_r.append(n_xc_after_x_integral)

    n_xc_after_x_integral_r = np.asarray(n_xc_after_x_integral_r)

    avg_xc_hole = np.pi * np.trapz(grids * grids * n * n_xc_after_x_integral_r,
                                   grids)

    return avg_xc_hole


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


# other
def txt_file_to_array(file, header=False):
    start = 0
    if header:
        start = 1
    # two column file to np arrays
    with open(file) as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines[start:]]
        y = [float(line.split()[1]) for line in lines[start:]]

    x = np.asarray(x)
    y = np.asarray(y)
    return x, y
