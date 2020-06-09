import matplotlib.pyplot as plt
import numpy as np
import sys
import two_el_exact
import ext_potentials
import blue_tools
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


run = 'M'
if run == 'n':
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

    r_s = (3 / (4 * np.pi * 1)) ** (1 / 3)
    avg_r_s = np.trapz((n_dmrg ** (2 / 3)) * r_s, grids) / 2

    # erf
    gam_files = ['r_s', 0, 1, 2, 3, 'inf']

    blue_CP_erf = []
    n_xc_blue_erf = []
    n_xc_blue_erf_interp = []
    for gam in gam_files:
        blue_CP_erf_gam = np.load(
            'erf_test/n_r0_1D_He_erf_gam_' + str(gam) + '.npy')
        blue_CP_erf.append(blue_CP_erf_gam)
        n_xc_blue_erf_gam = blue_CP_to_n_xc(blue_CP_erf_gam, n_dmrg)
        n_xc_blue_erf.append(n_xc_blue_erf_gam)
        n_xc_blue_erf_interp.append(get_interp_n_xc(grids, n_xc_blue_erf_gam))

    u_grids = copy.deepcopy(grids)
    avg_n_xc_exact = []
    avg_n_x_exact = []

    for u in u_grids:
        avg_n_xc_exact.append(get_avg_n_xc(u, grids, n_xc_exact_interp, n_dmrg))
        avg_n_x_exact.append(get_avg_n_xc(u, grids, n_x_exact_interp, n_dmrg))

    avg_n_xc_exact = np.asarray(avg_n_xc_exact)
    avg_n_x_exact = np.asarray(avg_n_x_exact)

    # erf interpolation
    avg_n_xc_blue_erf = []
    for i, gam in enumerate(gam_files):
        avg_n_xc_blue_erf_gam = []
        for u in u_grids:
            avg_n_xc_blue_erf_gam.append(
                get_avg_n_xc(u, grids, n_xc_blue_erf_interp[i], n_dmrg))
        avg_n_xc_blue_erf_gam = np.asarray(avg_n_xc_blue_erf_gam)
        avg_n_xc_blue_erf.append(avg_n_xc_blue_erf_gam)

    # symmetric plotting, |u|
    zero_u_idx = 256
    u_grids = u_grids[zero_u_idx:]
    avg_n_xc_exact = avg_n_xc_exact[zero_u_idx:]
    for i, gam in enumerate(gam_files):
        avg_n_xc_blue_erf[i] = avg_n_xc_blue_erf[i][zero_u_idx:]

    avg_n_x_exact = avg_n_x_exact[zero_u_idx:]

    # correlation
    avg_n_c_exact = avg_n_xc_exact - avg_n_x_exact
    avg_n_c_blue_erf = []
    for i, gam in enumerate(gam_files):
        avg_n_c_blue_erf.append(avg_n_xc_blue_erf[i] - avg_n_x_exact)


    # plots --------
    def do_plot():
        plt.xlabel('$|u|$', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(alpha=0.4)
        plt.show()


    gam_disp = ['1.5/r_s', 0, 1, 2, 3, '\infty']
    # avg n_xc
    for i, gam in enumerate(gam_disp):
        plt.plot(u_grids, avg_n_xc_blue_erf[i],
                 label=r'$\gamma = ' + str(
                     gam) + r'$')

    plt.plot(u_grids, avg_n_xc_exact,
             label=r'exact')
    do_plot()

    # avg_n_xc weighted with Vee
    for i, gam in enumerate(gam_disp):
        plt.plot(u_grids,
                 avg_n_xc_blue_erf[i] * -1 * ext_potentials.exp_hydrogenic(
                     u_grids),
                 label=r'$\gamma = ' + str(
                     gam) + r'$')

    plt.plot(u_grids, avg_n_xc_exact * -1 * ext_potentials.exp_hydrogenic(
        u_grids),
             label=r'exact')
    do_plot()

    # avg_n_c
    for i, gam in enumerate(gam_disp):
        plt.plot(u_grids, avg_n_c_blue_erf[i],
                 label=r'$\gamma = ' + str(
                     gam) + r'$')
    plt.plot(u_grids, avg_n_c_exact,
             label=r'exact')
    do_plot()

    # avg_n_c weighted with Vee
    for i, gam in enumerate(gam_disp):
        plt.plot(u_grids,
                 avg_n_c_blue_erf[i] * -1 * ext_potentials.exp_hydrogenic(
                     u_grids),
                 label=r'$\gamma = ' + str(
                     gam) + r'$')

    plt.plot(u_grids, avg_n_c_exact * -1 * ext_potentials.exp_hydrogenic(
        u_grids),
             label=r'exact')
    do_plot()


    # integrated plots, ie so the u->infty value matches the energy
    def integrated_avg_n_xc(u_grids, avg_n_xc_times_interaction):
        int_avg_n_xc = []
        for i, u in enumerate(u_grids):
            int_avg_n_xc.append(
                np.trapz(avg_n_xc_times_interaction[0:i], u_grids[0:i]))

        int_avg_n_xc = np.asarray(int_avg_n_xc)
        return int_avg_n_xc


    # xc plots
    for i, gam in enumerate(gam_disp):
        plt.plot(u_grids,
                 integrated_avg_n_xc(u_grids, avg_n_xc_blue_erf[
                     i] * -1 * ext_potentials.exp_hydrogenic(
                     u_grids)),
                 label=r'$\gamma = ' + str(
                     gam) + r'$')

    plt.plot(u_grids, integrated_avg_n_xc(u_grids,
                                          avg_n_xc_exact * -1 * ext_potentials.exp_hydrogenic(
                                              u_grids)),
             label=r'exact')
    do_plot()

    # c plots
    for i, gam in enumerate(gam_disp):
        plt.plot(u_grids,
                 integrated_avg_n_xc(u_grids, avg_n_c_blue_erf[
                     i] * -1 * ext_potentials.exp_hydrogenic(
                     u_grids)),
                 label=r'$\gamma = ' + str(
                     gam) + r'$')

    plt.plot(u_grids, integrated_avg_n_xc(u_grids,
                                          avg_n_c_exact * -1 * ext_potentials.exp_hydrogenic(
                                              u_grids)),
             label=r'exact')
    do_plot()

    # create table

    # exact
    U_xc_exact = np.trapz(avg_n_xc_exact * -1 * ext_potentials.exp_hydrogenic(
        u_grids), u_grids)

    U_c_exact = np.trapz(avg_n_c_exact * -1 * ext_potentials.exp_hydrogenic(
        u_grids), u_grids)

    for i, gam in enumerate(gam_disp):
        # method
        blue_tools.table_print('blue ($\gamma = ' + str(gam) + '$)')

        # U_xc
        U_xc_blue_gam = np.trapz(avg_n_xc_blue_erf[
                                     i] * -1 * ext_potentials.exp_hydrogenic(
            u_grids), u_grids)
        blue_tools.table_print(U_xc_blue_gam, round_to_dec=4)

        # U_c
        U_c_blue_gam = np.trapz(avg_n_c_blue_erf[
                                    i] * -1 * ext_potentials.exp_hydrogenic(
            u_grids), u_grids)
        blue_tools.table_print(U_c_blue_gam, round_to_dec=4)

        # U_c error
        U_c_error_gam = 100 * (U_c_exact - U_c_blue_gam) / U_c_exact
        blue_tools.table_print(U_c_error_gam, round_to_dec=1, last_in_row=True)

    blue_tools.table_print('exact')
    blue_tools.table_print(U_xc_exact, round_to_dec=4)
    blue_tools.table_print(U_c_exact, round_to_dec=4)
    blue_tools.table_print(0.0, round_to_dec=1, last_in_row=True)

    sys.exit()

if run == 'M':
    # plot M(r) measure, see logbook 5/25/20
    h = 0.08
    grids = np.arange(-256, 257) * h

    # exact ----------------
    P_r_rp_raw = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]
    P_r_rp = get_P_r_rp(P_r_rp_raw, n_dmrg, grids)
    n_x_exact = get_two_el_n_x(n_dmrg)
    n_xc_exact = pair_density_to_n_xc(P_r_rp, n_dmrg)

    # erf
    gam_files = ['r_s', 0, 1, 2, 3, 'inf']
    n_xc_blue_erf = []
    for gam in gam_files:
        blue_CP_erf_gam = np.load(
            'erf_test/n_r0_1D_He_erf_gam_' + str(gam) + '.npy')
        n_xc_blue_erf.append(blue_CP_to_n_xc(blue_CP_erf_gam, n_dmrg))

    # U_c plots -------------

    M_exact = get_M_measure(grids, n_xc_exact) - get_M_measure(grids, n_x_exact)
    M_blue_erf = [get_M_measure(grids, n_xc_blue_erf_gam) - get_M_measure(grids,
                                                                          n_x_exact)
                  for n_xc_blue_erf_gam in n_xc_blue_erf]

    int_Uc_blue_erf = [0.5 * M_blue_erf_gam * n_dmrg for M_blue_erf_gam in
                       M_blue_erf]
    int_Uc_exact = 0.5 * M_exact * n_dmrg

    gam_disp = ['1.5/r_s', 0, 1, 2, 3, '\infty']
    for i, gam in enumerate(gam_disp):
        plt.plot(grids, int_Uc_blue_erf[i],
                 label='$\gamma = ' + str(gam) + '$')
    plt.plot(grids, int_Uc_exact, label='exact,')
    plt.xlabel('$x$', fontsize=16)
    plt.xlim(-0.01, 5)

    do_plot()

    # U_xc plots -------------
    M_exact = get_M_measure(grids, n_xc_exact)
    M_blue_erf = [get_M_measure(grids, n_xc_blue_erf_gam) for n_xc_blue_erf_gam
                  in n_xc_blue_erf]

    int_Uxc_blue_erf = [0.5 * M_blue_erf_gam * n_dmrg for M_blue_erf_gam in
                        M_blue_erf]
    int_Uxc_exact = 0.5 * M_exact * n_dmrg

    Uxc_blue_erf = [np.trapz(int_Uxc_blue_erf_gam, grids) for
                    int_Uxc_blue_erf_gam
                    in int_Uxc_blue_erf]

    for i, gam in enumerate(gam_disp):
        plt.plot(grids, int_Uxc_blue_erf[i],
                 label='$\gamma = ' + str(gam) + '$')
    plt.plot(grids, int_Uxc_exact, label='exact')
    plt.xlabel('$x$', fontsize=16)
    plt.xlim(-0.01, 2)

    do_plot()

    sys.exit()

# (old) look at CP potential
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

    P_r_rp_idx = blue_tools.get_P_r_rp_idx(P_r_rp, n=n_dmrg, x_idx=x_idx,
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
