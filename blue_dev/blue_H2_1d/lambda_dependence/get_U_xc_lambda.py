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
    densities = np.load("../H2_data/densities.npy")

    # get stretched H2 values
    pot = potentials[50]
    n = densities[50]

    lambda_list = np.linspace(0, 1, 11)

    v_h = functionals.hartree_potential(grids,n)
    U = 0.5*np.sum(v_h*n)*h
    print('U = ', U)

    U_xc_lam = []
    for i, lam in enumerate(lambda_list):
        print("lam = ", lam)
        n2_r0 = np.load("n_r0_lambda.npy")[i]

        v_h_n_ee = functionals.get_v_n_xc(grids,n2_r0)
        V_ee_blue = 0.5*np.sum(v_h_n_ee*n)*h

        print("U_xc_blue(lam) = ", V_ee_blue - U)
        U_xc_lam.append(V_ee_blue - U)

    U_xc_lam = np.asarray(U_xc_lam)

    plt.plot(lambda_list, U_xc_lam)

    plt.ylabel('$U^B_{xc}(\lambda)$', fontsize=18)
    plt.xlabel('$\lambda$', fontsize=18)
    plt.grid(alpha=.4)
    plt.show()