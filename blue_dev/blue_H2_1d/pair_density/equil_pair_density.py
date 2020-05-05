import matplotlib.pyplot as plt
import numpy as np
import sys
import two_el_exact
import ext_potentials
import blue_pair_density

if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h

    # dmrg results
    P_r_rp = np.load('H2_equil_exact_pair_density.npy')
    n_dmrg = np.load('../H2_data/densities.npy')
    locations = np.load('../H2_data/locations.npy')

    # blue results
    blue_n2_r0 = np.load('../H2_data/n_r0_R.npy')

    # R = R_idx * h
    R_idx = 19
    n_dmrg = n_dmrg[R_idx]

    # blue_n2_r0 does not contain R = 0 result
    blue_n2_r0 = blue_n2_r0[R_idx - 1]

    x_value = 0.0
    x_idx = np.where(grids == x_value)[0][0]

    P_r_rp_idx = blue_pair_density.get_P_r_rp_idx(P_r_rp, n=n_dmrg, x_idx=x_idx,
                                                  h=h)

    print('n_dmrg[x_idx] ', n_dmrg[x_idx])
    print('integral check: n_dmrg = ', np.sum(n_dmrg) * h)

    print('integral check: P_r_rp_idx = ', np.sum(P_r_rp_idx) * h)

    # blue ----------------------------

    blue_n2_r0 = blue_n2_r0[x_idx]

    print('integral check: n2_r0 = ', np.sum(blue_n2_r0) * h)
    print('integral check: (P_r_rp_idx / n_dmrg[x_idx]) = ',
          np.sum((P_r_rp_idx / n_dmrg[x_idx])) * h)

    plt.plot(grids, (blue_n2_r0), label='$n^{Blue}_0(x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]),
             label='$P^{exact}(' + str(x_value) + ',x\prime)/n(' + str(
                 x_value) + ')$')
    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    plt.plot(grids, (blue_n2_r0) - n_dmrg,
             label='$n^{Blue}_{xc}(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]) - n_dmrg,
             label='$n_{xc}(' + str(x_value) + ',x\prime)$')
    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    plt.plot(grids, (blue_n2_r0) - n_dmrg / 2,
             label='$n^{Blue}_{c}(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]) - n_dmrg / 2,
             label='$n_{c}(' + str(x_value) + ',x\prime)$')
    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    # compare v_s of CP

    v_s_CP_blue = two_el_exact.v_s_extention(grids, blue_n2_r0, h)
    v_s_CP_exact = two_el_exact.v_s_extention(grids,
                                              (P_r_rp_idx / n_dmrg[x_idx]), h,
                                              tol=1.5 * (10 ** (-4)))
    plt.plot(grids, v_s_CP_blue,
             label='$v^{CP, Blue}_s(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, v_s_CP_exact,
             label='$v^{CP, Exact}_s(' + str(x_value) + ',x\prime)$')

    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()
