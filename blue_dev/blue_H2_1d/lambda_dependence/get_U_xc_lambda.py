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

if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h
    potentials = np.load("../H2_data/potentials.npy")
    dmrg_density = np.load("../H2_data/densities.npy")
    n_HF = np.load('n_HF.npy')

    # get stretched H2 values
    pot = potentials[50]
    n_dmrg = dmrg_density[50]

    lambda_list = np.linspace(0, 1, 11)

    v_h = functionals.hartree_potential(grids, n_HF)
    U_HF = 0.5 * np.sum(v_h * n_HF) * h
    print('Ex_HF = ', -U_HF/2)

    v_h_dmrg = functionals.hartree_potential(grids, n_dmrg)
    U_dmrg = 0.5 * np.sum(v_h_dmrg * n_dmrg) * h
    print('Ex_dmrg = ', -U_dmrg/2)

    U_xc_lam_HF = []
    U_xc_lam_dmrg = []
    for i, lam in enumerate(lambda_list):
        print("lam = ", lam)

        if lam == 0.0 and False:
            n2_r0_dmrg = np.load("n_r0_lambda_HF.npy")[i]

            plt.plot(grids, 2*n2_r0_dmrg[0], label=r'density from $ v_{ext} + (1/2)v_H[n^{DMRG}]$')
            plt.plot(grids, n_dmrg, label=r'$n^{DMRG}$')
            plt.legend()
            plt.show()


        # using HF density
        n2_r0_HF = np.load("n_r0_lambda_HF.npy")[i]
        v_h_n_HF_ee = functionals.get_v_n_xc(grids, n2_r0_HF)
        V_ee_blue_HF = 0.5 * np.sum(v_h_n_HF_ee * n_HF) * h

        # print("U_xc_blue(lam) = ", V_ee_blue_HF - U_HF)
        U_xc_lam_HF.append(V_ee_blue_HF - U_HF)

        # using dmrg density
        n2_r0_dmrg = np.load("n_r0_lambda.npy")[i]
        v_h_n_dmrg_ee = functionals.get_v_n_xc(grids, n2_r0_dmrg)
        V_ee_blue_dmrg = 0.5 * np.sum(v_h_n_dmrg_ee * n_dmrg) * h

        # print("U_xc_blue(lam) = ", V_ee_blue_HF - U_HF)
        U_xc_lam_dmrg.append(V_ee_blue_dmrg - U_dmrg)

    U_xc_lam_HF = np.asarray(U_xc_lam_HF)
    U_xc_lam_dmrg = np.asarray(U_xc_lam_dmrg)

    # HF values
    E_xc_HF = np.sum(U_xc_lam_HF) / (len(lambda_list))
    print('E_xc = ', E_xc_HF)

    E_c_HF = E_xc_HF - (-U_HF / 2)
    U_c_HF = U_xc_lam_HF[-1] - (-U_HF / 2)
    T_c_HF = E_c_HF - U_c_HF
    print('U_c_HF = ', U_c_HF)
    print('E_c_HF = ', E_c_HF)
    print('T_c_HF = ', T_c_HF)
    print()

    # DMRG values
    E_xc_dmrg = np.sum(U_xc_lam_dmrg) / (len(lambda_list))
    print('E_xc = ', E_xc_dmrg)

    E_c_dmrg = E_xc_dmrg - (-U_dmrg / 2)
    U_c_dmrg = U_xc_lam_dmrg[-1] - (-U_dmrg / 2)
    T_c_dmrg = E_c_dmrg - U_c_dmrg
    print('U_c_dmrg = ', U_c_dmrg)
    print('E_c_dmrg = ', E_c_dmrg)
    print('T_c_dmrg = ', T_c_dmrg)

    plt.plot(lambda_list, U_xc_lam_HF, label='using $n^{HF}$')
    plt.plot(lambda_list, U_xc_lam_dmrg, label='using $n^{DMRG}$')

    # plot our limits
    plt.scatter([0,1], [-U_dmrg/2, U_xc_lam_dmrg[-1]], color = 'red', label = '$E_x[n^{DMRG}]$ and $U^B_{xc}(\lambda = 1)[n^{DMRG}]$')


    plt.ylabel('$U^B_{xc}(\lambda)$', fontsize=18)
    plt.xlabel('$\lambda$', fontsize=18)
    plt.grid(alpha=.4)
    plt.legend(fontsize=16)
    plt.show()
