import sys
import numpy as np
import ext_potentials
import functionals
import matplotlib.pyplot as plt


def get_Uxc_lam(grids, n, n_r0_lambda, lambda_list):
    h = np.abs(grids[1] - grids[0])

    v_h = functionals.hartree_potential(grids, n)
    U = 0.5 * np.sum(v_h * n) * h

    U_xc_lam = []
    for i, lam in enumerate(lambda_list):
        # print("lam = ", lam)

        n2_r0 = n_r0_lambda[i]
        v_h_n_ee = functionals.get_v_n_xc(grids, n2_r0)
        V_ee_blue = 0.5 * np.sum(v_h_n_ee * n) * h

        U_xc_lam.append(V_ee_blue - U)

    U_xc_lam = np.asarray(U_xc_lam)

    return U, U_xc_lam


def get_Exc(grids, n, n_r0_lambda, lambda_list):
    U, U_xc_lam = get_Uxc_lam(grids, n, n_r0_lambda, lambda_list)
    E_xc = np.trapz(U_xc_lam, lambda_list)

    E_c = E_xc - U_xc_lam[0]
    U_c = U_xc_lam[-1] - U_xc_lam[0]
    T_c = E_c - U_c

    return U_c, T_c, E_c, E_xc
