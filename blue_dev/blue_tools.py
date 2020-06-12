import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import copy


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


def get_interp_n_xc(grids, n_xc):
    interp_n_xc = []
    for i, n_xc_r in enumerate(n_xc):
        # interpolated cubic spline function
        cubic_spline = InterpolatedUnivariateSpline(grids, n_xc_r, k=3)
        interp_n_xc.append(cubic_spline)

    interp_n_xc = np.asarray(interp_n_xc)
    return interp_n_xc


# 1d averaged symmetrized holes
def get_n_xc_sym(u, grids, n_xc_interp):
    n_xc_sym = []
    for i, x in enumerate(grids):
        n_xc_sym.append(0.5 * (n_xc_interp[i](u + x) + n_xc_interp[i](-u + x)))

    n_xc_sym = np.asarray(n_xc_sym)
    # division by zero: nan -> 0.0
    n_xc_sym = np.nan_to_num(n_xc_sym)
    return n_xc_sym


def get_avg_n_xc(u, grids, n_xc_interp, n):
    n_xc_sym_u = get_n_xc_sym(u, grids, n_xc_interp)
    avg_n_xc = np.trapz(n * n_xc_sym_u, grids)

    return avg_n_xc


def get_avg_sym_n_xc(grids, n, n_xc):
    n_xc_interp = get_interp_n_xc(grids, n_xc)

    u_grids = copy.deepcopy(grids)
    avg_n_xc = []
    for u in u_grids:
        avg_n_xc.append(get_avg_n_xc(u, grids, n_xc_interp, n))
    avg_n_xc = np.asarray(avg_n_xc)

    try:
        zero_u_idx = np.where(grids == 0.0)[0][0]
    except:
        print('error: no 0.0 in grids')
        return

    u_grids = u_grids[zero_u_idx:]
    avg_n_xc = avg_n_xc[zero_u_idx:]

    return u_grids, avg_n_xc


# tools for exact pair density

# output pair density 2 columns to 2d array
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


# 2d array raw pair density to exact pair density
def get_exact_pair_density(raw_pair_density, n, grids):
    h = np.abs(grids[1] - grids[0])

    P_r_rp = []
    for i, P_rp_raw in enumerate(raw_pair_density):
        P_rp = copy.deepcopy(P_rp_raw)
        P_rp[i] -= n[i] * h
        P_r_rp.append(P_rp / (h * h))

    P_r_rp = np.asarray(P_r_rp)
    return P_r_rp


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
