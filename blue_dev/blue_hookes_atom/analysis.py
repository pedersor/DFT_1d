import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import blue_tools


def get_n(r):
    # 3D hooke's atom radial density

    n = (2 / ((np.pi ** (3 / 2)) * (8 + 5 * (np.pi ** 0.5)))) * np.exp(
        -0.5 * (r ** 2)) * (((np.pi / 2) ** (0.5)) * (
            (7 / 4) + (0.25 * (r ** 2)) + (r + (1 / r)) * special.erf(
        r / np.sqrt(2))) + np.exp(-0.5 * (r ** 2)))

    n[0] = 0.0893193  # take the limit for r -> 0
    return n


if __name__ == '__main__':
    grids = np.linspace(0.0056, 6, 1000)

    n = get_n(grids)

    # sph blue results
    gammas = ['0', '0_43', '2', 'inf']
    n_r0_sph_blue = []
    v_H_n_ee_blue = []
    V_ee_blue = []
    for gam in gammas:
        n_r0_sph_blue_gam = np.load('n_r0_blue_gam_' + gam + '.npy')
        n_r0_sph_blue.append(n_r0_sph_blue_gam)

        v_H_n_ee_blue_gam = blue_tools.get_v_H_n_xc(grids, n_r0_sph_blue_gam)
        v_H_n_ee_blue.append(v_H_n_ee_blue_gam)
        V_ee_blue_gam = blue_tools.get_U_xc(grids, n, v_H_n_ee_blue_gam)
        V_ee_blue.append(V_ee_blue_gam)

    # sph exact results
    v_H_n = blue_tools.get_v_H_n(grids, n)
    U_H = blue_tools.get_U_xc(grids, n, v_H_n)
    E_x = -(U_H / 2)

    n_r0_sph_exact = np.load('n_r0_hookes_atom_sph_exact.npy')
    v_h_n_ee_sph = blue_tools.get_v_H_n_xc(grids, n_r0_sph_exact)
    V_ee_sph = blue_tools.get_U_xc(grids, n, v_h_n_ee_sph)
    U_xc_sph = V_ee_sph - U_H
    U_c_sph = V_ee_sph - U_H - E_x

    # non-sph exact results
    V_ee_exact = 0.447443
    U_xc_exact = V_ee_exact - U_H
    U_c_exact = V_ee_exact - U_H - E_x

    gammas = ['0', '0.43', '2', '\infty']
    # create table
    for i, V_ee_blue_gam in enumerate(V_ee_blue):
        # method
        blue_tools.table_print('blue ($\gamma = ' + gammas[i] + '$)')

        # U_xc
        U_xc_blue_gam = V_ee_blue_gam - U_H
        blue_tools.table_print(U_xc_blue_gam, round_to_dec=4)

        # U_c
        U_c_blue_gam = U_xc_blue_gam - E_x
        blue_tools.table_print(U_c_blue_gam, round_to_dec=4)

        # U_c sph exact error
        error_sph = 100 * (U_c_sph - U_c_blue_gam) / (U_c_sph)
        blue_tools.table_print(error_sph, round_to_dec=1)

        # U_c non-sph exact error
        error_sph = 100 * (U_c_exact - U_c_blue_gam) / (U_c_exact)
        blue_tools.table_print(error_sph, round_to_dec=1, last_in_row=True)

    # sph exact method
    blue_tools.table_print('sphericalized exact')
    blue_tools.table_print(U_xc_sph, round_to_dec=4)
    blue_tools.table_print(U_c_sph, round_to_dec=4)
    # U_c sph exact error
    blue_tools.table_print(0., round_to_dec=1)
    # U_c non-sph exact error
    error_sph = 100 * (U_c_exact - U_c_sph) / (U_c_exact)
    blue_tools.table_print(error_sph, round_to_dec=1, last_in_row=True)

    # non-sph exact method
    blue_tools.table_print('exact')
    blue_tools.table_print(U_xc_exact, round_to_dec=4)
    blue_tools.table_print(U_c_exact, round_to_dec=4)
    # U_c sph exact error
    error_sph = 100 * (U_c_sph - U_c_exact) / (U_c_sph)
    blue_tools.table_print(error_sph, round_to_dec=1)
    # U_c non-sph exact error
    blue_tools.table_print(0., round_to_dec=1, last_in_row=True)
