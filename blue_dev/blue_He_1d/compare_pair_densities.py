import matplotlib.pyplot as plt
import numpy as np
import sys
import two_el_exact
import ext_potentials
import blue_pair_density
import functionals
import copy

from scipy.interpolate import InterpolatedUnivariateSpline


# TODO: general, interpolate 2d array
# same from blue_He/xc_hole/plot_avg_xc_hole.py
def get_interp_n_xc(grids, n_xc):
    interp_n_xc = []
    for i, n_xc_r in enumerate(n_xc):
        # interpolated cubic spline function
        cubic_spline = InterpolatedUnivariateSpline(grids, n_xc_r, k=3)
        interp_n_xc.append(cubic_spline)

    interp_n_xc = np.asarray(interp_n_xc)
    return interp_n_xc


def pair_density_to_n_xc(pair_density, n):
    n_xc = []
    for i, P_rp in enumerate(pair_density):
        n_xc.append(-n + (P_rp / n[i]))
    n_xc = np.asarray(n_xc)
    # division by zero: nan -> 0.0
    n_xc = np.nan_to_num(n_xc)
    return n_xc


def blue_CP_to_n_xc(blue_CP, n):
    n_xc = []
    for i, CP_rp in enumerate(blue_CP):
        n_xc.append(-n + CP_rp)
    n_xc = np.asarray(n_xc)
    return n_xc


def get_two_el_n_x(n):
    n_x = []
    for i in range(len(n)):
        n_x.append(-n / 2.)
    n_x = np.asarray(n_x)
    return n_x


def get_P_r_rp(P_r_rp_raw, n, grids):
    h = np.abs(grids[1] - grids[0])

    P_r_rp = []
    for i, P_rp_raw in enumerate(P_r_rp_raw):
        P_rp = copy.deepcopy(P_rp_raw)
        P_rp[i] -= n[i] * h
        P_r_rp.append(P_rp / (h * h))

    P_r_rp = np.asarray(P_r_rp)
    return P_r_rp


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


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h

    # (x = 0) or specific values
    # exact ----------------
    P_r_rp_raw = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]
    P_r_rp = get_P_r_rp(P_r_rp_raw, n_dmrg, grids)
    n_x_exact = get_two_el_n_x(n_dmrg)
    n_x_exact_interp = get_interp_n_xc(grids, n_x_exact)

    n_xc_exact = pair_density_to_n_xc(P_r_rp, n_dmrg)
    n_xc_exact_interp = get_interp_n_xc(grids, n_xc_exact)

    # blue ------
    blue_CP = np.load('n_r0_0.npy')[0]
    n_xc_blue = blue_CP_to_n_xc(blue_CP, n_dmrg)
    n_xc_blue_interp = get_interp_n_xc(grids, n_xc_blue)

    u_grids = copy.deepcopy(grids)
    avg_n_xc_exact = []
    avg_n_x_exact = []
    avg_n_xc_blue = []
    for u in u_grids:
        avg_n_xc_exact.append(get_avg_n_xc(u, grids, n_xc_exact_interp, n_dmrg))
        avg_n_xc_blue.append(get_avg_n_xc(u, grids, n_xc_blue_interp, n_dmrg))
        avg_n_x_exact.append(get_avg_n_xc(u, grids, n_x_exact_interp, n_dmrg))

    avg_n_xc_exact = np.asarray(avg_n_xc_exact)
    avg_n_xc_blue = np.asarray(avg_n_xc_blue)
    avg_n_x_exact = np.asarray(avg_n_x_exact)

    '''
    # E_x check..
    E_x = np.trapz(
        avg_n_x_exact[256:] * -1 * ext_potentials.exp_hydrogenic(u_grids[256:]), u_grids[256:])
    print('E_x = ', E_x)

    v_H = functionals.hartree_potential(grids, n_dmrg)
    U = 0.5*np.trapz(v_H*n_dmrg, grids)
    print('E_x = -U/2 = ', -U/2.)
    '''

    # symmetric plotting, |u|
    zero_u_idx = 256
    u_grids = u_grids[zero_u_idx:]
    avg_n_xc_blue = avg_n_xc_blue[zero_u_idx:]
    avg_n_xc_exact = avg_n_xc_exact[zero_u_idx:]
    avg_n_x_exact = avg_n_x_exact[zero_u_idx:]

    # correlation
    avg_n_c_exact = avg_n_xc_exact - avg_n_x_exact
    avg_n_c_blue = avg_n_xc_blue - avg_n_x_exact

    # avg_n_xc
    # plt.plot(u_grids, avg_n_xc_blue, label='blue')
    # plt.plot(u_grids, avg_n_xc_exact, label='exact')

    # avg_n_xc weighted with Vee
    plt.plot(u_grids, avg_n_xc_blue * -1 * ext_potentials.exp_hydrogenic(
        u_grids), label='blue')
    plt.plot(u_grids, avg_n_xc_exact * -1 * ext_potentials.exp_hydrogenic(
        u_grids), label='exact')
    plt.show()

    # avg_n_c weighted with Vee
    plt.plot(u_grids, avg_n_c_blue * -1 * ext_potentials.exp_hydrogenic(
        u_grids), label='blue')
    plt.plot(u_grids, avg_n_c_exact * -1 * ext_potentials.exp_hydrogenic(
        u_grids), label='exact')

    # U_c check:
    print('U_c_blue = ',
          np.trapz(avg_n_c_blue * -1 * ext_potentials.exp_hydrogenic(
              u_grids), u_grids))
    print('U_c_exact = ',
          np.trapz(avg_n_c_exact * -1 * ext_potentials.exp_hydrogenic(
              u_grids), u_grids))

    plt.show()

    sys.exit()

    x_value = 0.0
    x_idx = np.where(grids == x_value)[0][0]

    P_r_rp_idx = blue_pair_density.get_P_r_rp_idx(P_r_rp_raw, n=n_dmrg,
                                                  x_idx=x_idx,
                                                  h=h)

    # blue ----------------------------
    blue_CP = np.load('n_r0_0.npy')[0][x_idx]

    plt.plot(grids, n_xc_blue_interp[x_idx](grids),
             label='$n^{Blue}_{xc}(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, n_xc_exact_interp[x_idx](grids),
             label='$n_{xc}(' + str(x_value) + ',x\prime)$')
    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    plt.plot(grids, (blue_CP) - n_dmrg / 2,
             label='$n^{Blue}_{c}(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, n_xc_exact[x_idx] + n_dmrg / 2,
             label='$n_{c}(' + str(x_value) + ',x\prime)$')
    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    # compare v_s of CP

    v_s_CP_blue = two_el_exact.v_s_extention(grids, blue_CP, h)
    v_s_CP_exact = two_el_exact.v_s_extention(grids,
                                              (P_r_rp_idx / n_dmrg[x_idx]), h,
                                              tol=1.1 * (10 ** (-4)))
    plt.plot(grids, v_s_CP_blue,
             label='$v^{CP, Blue}_s(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, v_s_CP_exact,
             label='$v^{CP, Exact}_s(' + str(x_value) + ',x\prime)$')

    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    sys.exit()
    if __name__ == '__main__':
        h = 0.08
    grids = np.arange(-256, 257) * h

    # (x = 0) or specific values
    # exact ----------------
    P_r_rp = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]

    x_value = 1.04
    x_idx = np.where(grids == x_value)[0][0]
    print(x_idx)

    P_r_rp_idx = blue_pair_density.get_P_r_rp(P_r_rp, n=n_dmrg, x_idx=x_idx,
                                              h=h)

    print('n_dmrg[x_idx] ', n_dmrg[x_idx])
    print('integral check: n_dmrg = ', np.sum(n_dmrg) * h)

    print('integral check: P_r_rp_idx = ', np.sum(P_r_rp_idx) * h)

    # blue ----------------------------

    blue_CP = np.load('n_r0_0.npy')[0][x_idx]

    print('integral check: n2_r0 = ', np.sum(blue_CP) * h)
    print('integral check: (P_r_rp_idx / n_dmrg[x_idx]) = ',
          np.sum((P_r_rp_idx / n_dmrg[x_idx])) * h)

    plt.plot(grids, (blue_CP), label='$n^{Blue}_0(x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]),
             label='$P^{exact}(' + str(x_value) + ',x\prime)/n(' + str(
                 x_value) + ')$')
    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    plt.plot(grids, (blue_CP) - n_dmrg,
             label='$n^{Blue}_{xc}(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]) - n_dmrg,
             label='$n_{xc}(' + str(x_value) + ',x\prime)$')
    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    plt.plot(grids, (blue_CP) - n_dmrg / 2,
             label='$n^{Blue}_{c}(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]) - n_dmrg / 2,
             label='$n_{c}(' + str(x_value) + ',x\prime)$')
    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    # compare v_s of CP

    v_s_CP_blue = two_el_exact.v_s_extention(grids, blue_CP, h)
    v_s_CP_exact = two_el_exact.v_s_extention(grids,
                                              (P_r_rp_idx / n_dmrg[x_idx]), h,
                                              tol=1.1 * (10 ** (-4)))
    plt.plot(grids, v_s_CP_blue,
             label='$v^{CP, Blue}_s(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, v_s_CP_exact,
             label='$v^{CP, Exact}_s(' + str(x_value) + ',x\prime)$')

    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()
