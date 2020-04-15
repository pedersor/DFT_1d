import sys
import numpy as np
import ext_potentials
import functionals
import matplotlib.pyplot as plt

# plotting parameters
params = {'mathtext.default': 'default'}
plt.rcParams.update(params)
plt.rcParams['axes.axisbelow'] = True
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
fig, ax = plt.subplots()


def get_Uxc_lam(grids, n, n_r0_lambda, lambda_list):
    h = np.abs(grids[1] - grids[0])

    v_h = functionals.hartree_potential(grids, n)
    U = 0.5 * np.sum(v_h * n) * h

    U_xc_lam = []
    for i, lam in enumerate(lambda_list):
        #print("lam = ", lam)

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


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h
    potentials = np.load("../H2_data/potentials.npy")
    dmrg_density = np.load("../H2_data/densities.npy")
    n_HF = np.load('n_HF.npy')

    # get stretched H2 values
    pot = potentials[50]
    n_dmrg = dmrg_density[50]

    n_r0_lambda_HF = np.load("n_r0_lambda_HF.npy")
    n2_r0_lambda_dmrg = np.load("n_r0_lambda.npy")

    lambda_list = np.linspace(0, 1, 11)

    U_HF, U_xc_HF = get_Uxc_lam(grids, n_HF, n_r0_lambda_HF, lambda_list)
    U_dmrg, U_xc_dmrg = get_Uxc_lam(grids, n_dmrg, n2_r0_lambda_dmrg,
                                    lambda_list)

    U_c_HF, T_c_HF, E_c_HF, E_xc_HF = get_Exc(grids, n_HF, n_r0_lambda_HF,
                                              lambda_list)

    print('U_c_HF = ', U_c_HF)
    print('T_c_HF = ', T_c_HF)
    print('E_c_HF = ', E_c_HF)
    print('E_xc_HF = ', E_xc_HF)
    print()

    U_c_dmrg, T_c_dmrg, E_c_dmrg, E_xc_dmrg = get_Exc(grids, n_dmrg,
                                                      n2_r0_lambda_dmrg,
                                                      lambda_list)

    print('U_c_dmrg = ', U_c_dmrg)
    print('T_c_dmrg = ', T_c_dmrg)
    print('E_c_dmrg = ', E_c_dmrg)
    print('E_xc_dmrg = ', E_xc_dmrg)

    plt.plot(lambda_list, U_xc_HF, label='using $n^{HF}$')
    plt.plot(lambda_list, U_xc_dmrg, label='using $n^{DMRG}$')

    # plot our limits
    plt.scatter([0, 1], [-U_dmrg / 2, U_xc_dmrg[-1]], color='red',
                label='$E_x[n^{DMRG}]$ and $U^B_{xc}(\lambda = 1)[n^{DMRG}]$')

    plt.ylabel('$U^B_{xc}(\lambda)$', fontsize=18)
    plt.xlabel('$\lambda$', fontsize=18)
    plt.grid(alpha=.4)
    plt.legend(fontsize=16)
    plt.show()
