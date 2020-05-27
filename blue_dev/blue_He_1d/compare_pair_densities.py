import matplotlib.pyplot as plt
import numpy as np
import sys
import two_el_exact
import ext_potentials
import blue_pair_density
import single_electron
import functionals
import functools
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


def do_plot():
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()


def get_M_measure(grids, n_xc):
    # M(r) measure, see logbook 5/25/20
    M = []
    for i, n_xc_r in enumerate(n_xc):
        M_r = np.trapz(
            n_xc_r * -1 * ext_potentials.exp_hydrogenic(grids - grids[i]),
            grids)
        M.append(M_r)
    M = np.asarray(M)
    return M


if __name__ == '__main__':
    # plot M(r) measure, see logbook 5/25/20
    h = 0.08
    grids = np.arange(-256, 257) * h

    # exact ----------------
    P_r_rp_raw = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]
    P_r_rp = get_P_r_rp(P_r_rp_raw, n_dmrg, grids)
    n_x_exact = get_two_el_n_x(n_dmrg)
    n_xc_exact = pair_density_to_n_xc(P_r_rp, n_dmrg)

    # blue ------
    blue_CP = np.load('n_r0_0.npy')[0]
    n_xc_blue = blue_CP_to_n_xc(blue_CP, n_dmrg)
    # e/2 charge
    blue_CP_half = np.load('n_r0_1D_He_half.npy')[0]
    n_xc_blue_half = blue_CP_to_n_xc(blue_CP_half, n_dmrg)

    # U_c plots -------------

    M_exact = get_M_measure(grids, n_xc_exact) - get_M_measure(grids, n_x_exact)
    M_blue = get_M_measure(grids, n_xc_blue) - get_M_measure(grids, n_x_exact)
    M_blue_half = get_M_measure(grids, n_xc_blue_half) - get_M_measure(grids,
                                                                       n_x_exact)

    int_Uc_blue_half = 0.5 * M_blue_half * n_dmrg
    int_Uc_blue = 0.5 * M_blue * n_dmrg
    int_Uc_exact = 0.5 * M_exact * n_dmrg

    Uc_blue_half = np.trapz(int_Uc_blue_half, grids)
    Uc_blue = np.trapz(int_Uc_blue, grids)
    Uc_exact = np.trapz(int_Uc_exact, grids)
    print('Uc_blue_half = ', Uc_blue_half)
    print('Uc_blue = ', Uc_blue)
    print('Uc_exact = ', Uc_exact)

    plt.plot(grids, int_Uc_blue_half,
             label='blue ($e^B = 1/2$), $U^B_c = $' + format(Uc_blue_half,
                                                             '.4f'))
    plt.plot(grids, int_Uc_blue,
             label='blue ($e^B = 1$), $U^B_c = $' + format(Uc_blue, '.4f'))
    plt.plot(grids, int_Uc_exact,
             label='exact, $U_c = $' + format(Uc_exact, '.4f'))
    plt.xlabel('$x$', fontsize=16)
    plt.xlim(-0.01, 6)

    do_plot()

    # U_xc plots -------------
    M_exact = get_M_measure(grids, n_xc_exact)
    M_blue = get_M_measure(grids, n_xc_blue)
    M_blue_half = get_M_measure(grids, n_xc_blue_half)

    int_Uxc_blue_half = 0.5 * M_blue_half * n_dmrg
    int_Uxc_blue = 0.5 * M_blue * n_dmrg
    int_Uxc_exact = 0.5 * M_exact * n_dmrg

    Uxc_blue_half = np.trapz(int_Uxc_blue_half, grids)
    Uxc_blue = np.trapz(int_Uxc_blue, grids)
    Uxc_exact = np.trapz(int_Uxc_exact, grids)
    print('Uxc_blue_half = ', Uxc_blue_half)
    print('Uxc_blue = ', Uxc_blue)
    print('Uxc_exact = ', Uxc_exact)
    print('Ex = ', Uxc_exact - Uc_exact)

    plt.plot(grids, int_Uxc_blue_half,
             label='blue ($e^B = 1/2$), $U^B_{xc} = $' + format(Uxc_blue_half,
                                                             '.4f'))
    plt.plot(grids, int_Uxc_blue,
             label='blue ($e^B = 1$), $U^B_{xc} = $' + format(Uxc_blue, '.4f'))
    plt.plot(grids, int_Uxc_exact,
             label='exact, $U_{xc} = $' + format(Uxc_exact, '.4f'))
    plt.xlabel('$x$', fontsize=16)
    plt.xlim(-0.01, 6)

    do_plot()

    sys.exit()

if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h

    # (x = 0) or specific values
    # exact ----------------
    P_r_rp = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]

    x_value = -0.72
    x_idx = np.where(grids == x_value)[0][0]
    print(x_idx)

    P_r_rp_idx = blue_pair_density.get_P_r_rp_idx(P_r_rp, n=n_dmrg, x_idx=x_idx,
                                                  h=h)

    print('n_dmrg[x_idx] ', n_dmrg[x_idx])
    print('integral check: n_dmrg = ', np.sum(n_dmrg) * h)

    print('integral check: P_r_rp_idx = ', np.sum(P_r_rp_idx) * h)

    # blue ----------------------------

    blue_CP = np.load('n_r0_0.npy')[0][x_idx]
    # e/2 charge
    blue_CP = np.load('n_r0_1D_He_half.npy')[0][x_idx]

    print('integral check: n2_r0 = ', np.sum(blue_CP) * h)
    print('integral check: (P_r_rp_idx / n_dmrg[x_idx]) = ',
          np.sum((P_r_rp_idx / n_dmrg[x_idx])) * h)

    plt.plot(grids, (blue_CP), label='$n^{Blue}_0(x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]),
             label='$P^{exact}(' + str(x_value) + ',x\prime)/n(' + str(
                 x_value) + ')$')
    plt.xlabel('$x\prime$', fontsize=16)
    do_plot()

    plt.plot(grids, (blue_CP) - n_dmrg,
             label='$n^{Blue}_{xc}(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]) - n_dmrg,
             label='$n_{xc}(' + str(x_value) + ',x\prime)$')
    plt.xlabel('$x\prime$', fontsize=16)
    do_plot()

    plt.plot(grids, (blue_CP) - n_dmrg / 2,
             label='$n^{Blue}_{c}(' + str(x_value) + ',x\prime)$')
    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]) - n_dmrg / 2,
             label='$n_{c}(' + str(x_value) + ',x\prime)$')
    plt.xlabel('$x\prime$', fontsize=16)
    do_plot()

    # compare v_s of CP
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')

    ax1.plot(grids, blue_CP)
    ax1.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]))

    # with extension
    v_s_CP_blue = two_el_exact.v_s_extension(grids, blue_CP, h)
    v_s_CP_exact = two_el_exact.v_s_extension(grids,
                                              (P_r_rp_idx / n_dmrg[x_idx]), h,
                                              tol=1. * (10 ** (-4)))

    ax2.plot(grids, v_s_CP_blue - ext_potentials.exp_hydrogenic(grids, Z=2),
             label='$v^{CP, Blue}_s(' + str(
                 x_value) + ',x\prime) - v(x\prime)$')
    ax2.plot(grids, v_s_CP_exact - ext_potentials.exp_hydrogenic(grids, Z=2),
             label='$v^{CP, Exact}_s(' + str(
                 x_value) + ',x\prime) - v(x\prime)$')

    plt.xlabel('$x\prime$', fontsize=16)
    plt.legend(fontsize=14)
    plt.xlim(-10, 10)

    ax1.grid(alpha=0.4)
    ax2.grid(alpha=0.4)
    plt.show()

    # check complimentary condition ----------

    solver = single_electron.EigenSolver(grids,
                                         potential_fn=functools.partial(
                                             ext_potentials.get_gridded_potential,
                                             potential=v_s_CP_exact),
                                         boundary_condition='open',
                                         num_electrons=1)
    solver.solve_ground_state()
    eps_r = solver.eigenvalues[0]
    print('eps_r = ', eps_r)

    idx_0 = 256
    idx_104 = 269
    idx_run = idx_0

    v_ext_0 = ext_potentials.exp_hydrogenic(0.0, Z=2)
    v_ext_104 = ext_potentials.exp_hydrogenic(1.04, Z=2)
    E = -2.237

    print('v_s_CP: ')
    print(v_s_CP_exact[idx_run])

    print('v_s_CP - v_ext')
    print(v_s_CP_exact[idx_run] - v_ext_104)

    print('v_ee(r-rp) - E + eps_r + eps_rp = ',
          -1 * ext_potentials.exp_hydrogenic(1.04) - E + (-0.76955 - 0.7741))

    sys.exit()

if __name__ == '__main__':
    # symmetrized plotting <n_c(u)> etc.
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
    # e/2 charge
    blue_CP = np.load('n_r0_1D_He_half.npy')[0]
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

    # symmetric plotting, |u|
    zero_u_idx = 256
    u_grids = u_grids[zero_u_idx:]
    avg_n_xc_blue = avg_n_xc_blue[zero_u_idx:]
    avg_n_xc_exact = avg_n_xc_exact[zero_u_idx:]
    avg_n_x_exact = avg_n_x_exact[zero_u_idx:]

    # correlation
    avg_n_c_exact = avg_n_xc_exact - avg_n_x_exact
    avg_n_c_blue = avg_n_xc_blue - avg_n_x_exact


    # plots --------
    def do_plot():
        plt.xlabel('$|u|$', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(alpha=0.4)
        plt.show()


    # avg_n_xc
    plt.plot(u_grids, avg_n_xc_blue,
             label=r'$\left\langle n^{Blue}_{xc}(|u|) \right\rangle$')
    plt.plot(u_grids, avg_n_xc_exact,
             label=r'$\left\langle n^{Exact}_{xc}(|u|) \right\rangle$')
    do_plot()

    # avg_n_xc weighted with Vee
    plt.plot(u_grids, avg_n_xc_blue * -1 * ext_potentials.exp_hydrogenic(
        u_grids),
             label=r'$\left\langle n^{Blue}_{xc}(|u|) \right\rangle v_{ee}(|u|)$')
    plt.plot(u_grids, avg_n_xc_exact * -1 * ext_potentials.exp_hydrogenic(
        u_grids),
             label=r'$\left\langle n^{Exact}_{xc}(|u|) \right\rangle v_{ee}(|u|)$')
    do_plot()

    # avg_n_c
    plt.plot(u_grids, avg_n_c_blue,
             label=r'$\left\langle n^{Blue}_{c}(|u|) \right\rangle$')
    plt.plot(u_grids, avg_n_c_exact,
             label=r'$\left\langle n^{Exact}_{c}(|u|) \right\rangle$')
    do_plot()

    # avg_n_c weighted with Vee
    plt.plot(u_grids, avg_n_c_blue * -1 * ext_potentials.exp_hydrogenic(
        u_grids),
             label=r'$\left\langle n^{Blue}_{c}(|u|) \right\rangle v_{ee}(|u|)$')
    plt.plot(u_grids, avg_n_c_exact * -1 * ext_potentials.exp_hydrogenic(
        u_grids),
             label=r'$\left\langle n^{Exact}_{c}(|u|) \right\rangle v_{ee}(|u|)$')
    do_plot()

    # U_c check:
    print('U_c_blue = ',
          np.trapz(avg_n_c_blue * -1 * ext_potentials.exp_hydrogenic(
              u_grids), u_grids))
    print('U_c_exact = ',
          np.trapz(avg_n_c_exact * -1 * ext_potentials.exp_hydrogenic(
              u_grids), u_grids))

    # E_x check:
    E_x = np.trapz(
        avg_n_x_exact * -1 * ext_potentials.exp_hydrogenic(u_grids), u_grids)
    print('E_x = ', E_x)

    v_H = functionals.hartree_potential(grids, n_dmrg)
    U = 0.5 * np.trapz(v_H * n_dmrg, grids)
    print('E_x = -U/2 = ', -U / 2.)

    sys.exit()
