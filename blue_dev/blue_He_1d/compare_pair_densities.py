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


def get_T_mat(grids):
    solver = single_electron.EigenSolver(grids,
                                         potential_fn=functools.partial(
                                             ext_potentials.exp_hydrogenic),
                                         boundary_condition='open',
                                         num_electrons=1)

    T_mat = solver.get_kinetic_matrix()
    return T_mat


def get_T_from_pair_density(grids, pair_density):
    Psi = []
    for P_x1 in pair_density:
        Psi_x1 = np.sqrt(np.abs(P_x1) / 2)
        Psi.append(Psi_x1)
    Psi = np.asarray(Psi)

    # kinetic energy operator matrix, T_mat
    T_mat = get_T_mat(grids)

    # T_mat_x2 Psi(x1,x2)
    T_mat_x2_Psi = []
    for Psi_x1 in Psi:
        T_mat_x2_Psi_x1 = np.matmul(T_mat, Psi_x1)
        T_mat_x2_Psi.append(T_mat_x2_Psi_x1)
    T_mat_x2_Psi = np.asarray(T_mat_x2_Psi)

    # T_mat_x1 Psi(x1, x2)
    T_mat_x1_Psi = []
    for Psi_x2 in np.transpose(Psi):
        T_mat_x1_Psi_x2 = np.matmul(T_mat, Psi_x2)
        T_mat_x1_Psi.append(T_mat_x1_Psi_x2)
    T_mat_x1_Psi = np.asarray(T_mat_x1_Psi)
    T_mat_x1_Psi = np.transpose(T_mat_x1_Psi)

    # \int dx_2 Psi(x1, x2)*( T_mat_x1 Psi(x1, x2) + T_mat_x2 Psi(x1, x2))
    integral_1 = []
    for i, Psi_x1 in enumerate(Psi):
        int_dx1 = Psi_x1 * (T_mat_x1_Psi[i] + T_mat_x2_Psi[i])
        integral_1.append(np.trapz(int_dx1, grids))
    integral_1 = np.asarray(integral_1)

    T = np.trapz(integral_1, grids)

    return T


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


class Quantities():
    def __init__(self, label, n_x, n_xc, pair_density=None):
        self.label = label
        self.n_x = n_x
        self.n_xc = n_xc
        self.pair_density = pair_density

    def get_T(self, grids):
        return get_T_from_pair_density(grids, self.pair_density)

    def get_U_xc(self, grids, n):
        u_grids, avg_sym_n_xc = get_avg_sym_n_xc(grids, n, self.n_xc)

        U_xc = np.trapz(avg_sym_n_xc * -1 * ext_potentials.exp_hydrogenic(
            u_grids), u_grids)
        return U_xc

    def get_U_c(self, grids, n):
        u_grids, avg_sym_n_xc = get_avg_sym_n_xc(grids, n, self.n_xc)
        u_grids, avg_sym_n_x = get_avg_sym_n_xc(grids, n, self.n_x)
        avg_sym_n_c = avg_sym_n_xc - avg_sym_n_x

        U_c = np.trapz(avg_sym_n_c * -1 * ext_potentials.exp_hydrogenic(
            u_grids), u_grids)

        return U_c


run = 'kin'
gam_files = ['1or_s', '3o2r_s', 0, 2, 3, 'inf']
gam_disp = ['1/r_s', '1.5/r_s', 0, 2, 3, '\infty']

# kinetic energy from pair density
if run == 'kin':
    h = 0.08
    grids = np.arange(-256, 257) * h

    to_compare = []

    # exact ----------------
    E_c_exact = -0.014
    T_s_exact = 0.273
    P_r_rp_raw = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]
    pair_density = get_P_r_rp(P_r_rp_raw, n_dmrg, grids)
    n_x_exact = get_two_el_n_x(n_dmrg)
    n_xc_exact = pair_density_to_n_xc(pair_density, n_dmrg)

    # blue -----------------
    gam_files = ['1or_s', '3o2r_s', '0', '1', '2', '3', 'inf']
    gam_labels = ['1/r_s  ', '3/(2r_s)  ', '0 ', '1 ', '2', '3', r'\infty ']
    for i, gam in enumerate(gam_files):
        blue_CP = np.load('erf_test/n_r0_1D_He_erf_gam_' + str(gam) + '.npy')

        blue_pair_density = []
        for j, blue_CP_x1 in enumerate(blue_CP):
            blue_pair_density.append(n_dmrg[j] * blue_CP_x1)
        blue_pair_density = np.asarray(blue_pair_density)
        n_xc_blue = blue_CP_to_n_xc(blue_CP, n_dmrg)

        gam_label = 'blue, $\gamma = ' + gam_labels[i] + '$'

        Q_blue = Quantities(label=gam_label, n_x=n_x_exact,
                            n_xc=n_xc_blue, pair_density=blue_pair_density)
        to_compare.append(Q_blue)

    # append exact quantities
    Q_exact = Quantities(label='exact', n_x=n_x_exact, n_xc=n_xc_exact,
                         pair_density=pair_density)
    to_compare.append(Q_exact)

    for method in to_compare:
        # method
        blue_tools.table_print(method.label)

        # U_xc
        blue_tools.table_print(method.get_U_xc(grids, n_dmrg))

        # U_c
        U_c_method = method.get_U_c(grids, n_dmrg)
        blue_tools.table_print(U_c_method)

        # T_c
        T_c_method = method.get_T(grids) - T_s_exact
        blue_tools.table_print(T_c_method)

        # E_c
        E_c_method = U_c_method + T_c_method
        blue_tools.table_print(E_c_method, last_in_row=True)

    sys.exit()

# symmetrized plotting <n_c(u)> etc.
if run == 'n':
    h = 0.08
    grids = np.arange(-256, 257) * h

    # (x = 0) or specific values
    # exact ----------------
    P_r_rp_raw = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]
    pair_density = get_P_r_rp(P_r_rp_raw, n_dmrg, grids)
    n_x_exact = get_two_el_n_x(n_dmrg)
    n_x_exact_interp = get_interp_n_xc(grids, n_x_exact)

    n_xc_exact = pair_density_to_n_xc(pair_density, n_dmrg)
    n_xc_exact_interp = get_interp_n_xc(grids, n_xc_exact)

    r_s = (3 / (4 * np.pi * 1)) ** (1 / 3)
    avg_r_s = np.trapz((n_dmrg ** (2 / 3)) * r_s, grids) / 2

    # erf

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

# plot M(r) measure, see logbook 5/25/20
if run == 'M':
    h = 0.08
    grids = np.arange(-256, 257) * h

    # exact ----------------
    P_r_rp_raw = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]
    pair_density = get_P_r_rp(P_r_rp_raw, n_dmrg, grids)
    n_x_exact = get_two_el_n_x(n_dmrg)
    n_xc_exact = pair_density_to_n_xc(pair_density, n_dmrg)

    # erf
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
