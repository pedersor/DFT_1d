import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import blue_tools
import sys


def get_n(r):
    # 3D hooke's atom radial density

    n = (2 / ((np.pi ** (3 / 2)) * (8 + 5 * (np.pi ** 0.5)))) * np.exp(
        -0.5 * (r ** 2)) * (((np.pi / 2) ** (0.5)) * (
            (7 / 4) + (0.25 * (r ** 2)) + (r + (1 / r)) * special.erf(
        r / np.sqrt(2))) + np.exp(-0.5 * (r ** 2)))

    n[0] = 0.0893193  # take the limit for r -> 0
    return n


def get_pair_density(r1, r2, theta):
    psi = (1 + 0.5 * np.sqrt(
        r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(theta))) / (2. * np.exp(
        0.25 * (r1 ** 2 + r2 ** 2)) * np.sqrt(
        8 * np.pi ** 2.5 + 5 * np.pi ** 3))

    return 2 * (psi ** 2)


gammas_files = ['0', '0_43', '0_613', '1_5', '2', 'inf']
gammas_disp = ['0', '0.43', r'1.5/\left\langle r_s \right\rangle = 0.613',
               '1.5', '2', '\infty']
run = 'M(r)'

# get table results U_xc, U_c, etc.
if run == 'table':
    grids = np.linspace(0.0056, 6, 1000)

    n = get_n(grids)

    # sph blue results
    n_r0_sph_blue = []
    v_H_n_ee_blue = []
    V_ee_blue = []
    for gam in gammas_files:
        n_r0_sph_blue_gam = np.load('n_r0_blue_gam_' + gam + '.npy')
        n_r0_sph_blue.append(n_r0_sph_blue_gam)

        v_H_n_ee_blue_gam = blue_tools.radial_get_v_H_n_xc(grids,
                                                           n_r0_sph_blue_gam)
        v_H_n_ee_blue.append(v_H_n_ee_blue_gam)
        V_ee_blue_gam = blue_tools.radial_get_U_xc(grids, n, v_H_n_ee_blue_gam)
        V_ee_blue.append(V_ee_blue_gam)

    # sph exact results
    v_H_n = blue_tools.radial_get_v_H_n(grids, n)
    U_H = blue_tools.radial_get_U_xc(grids, n, v_H_n)
    E_x = -(U_H / 2)

    n_r0_sph_exact = np.load('n_r0_hookes_atom_sph_exact.npy')
    v_h_n_ee_sph = blue_tools.radial_get_v_H_n_xc(grids, n_r0_sph_exact)
    V_ee_sph = blue_tools.radial_get_U_xc(grids, n, v_h_n_ee_sph)
    U_xc_sph = V_ee_sph - U_H
    U_c_sph = V_ee_sph - U_H - E_x

    # non-sph exact results
    V_ee_exact = 0.447443
    U_xc_exact = V_ee_exact - U_H
    U_c_exact = V_ee_exact - U_H - E_x

    # create table
    for i, V_ee_blue_gam in enumerate(V_ee_blue):
        # method
        blue_tools.table_print('blue ($\gamma = ' + gammas_disp[i] + '$)')

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

    sys.exit()

# get <n_xc(u)> plots
if run == 'n_xc':
    grids = np.linspace(0.0056, 6, 1000)

    n = get_n(grids)

    # sph blue results
    interp_n_xc_sph_blue = []
    for gam in gammas_files:
        n_r0_sph_blue_gam = np.load('n_r0_blue_gam_' + gam + '.npy')
        n_xc_sph_blue_gam = blue_tools.n_CP_to_n_xc(n_r0_sph_blue_gam, n)
        interp_n_xc_sph_blue_gam = blue_tools.get_interp_n_xc(grids,
                                                              n_xc_sph_blue_gam)
        interp_n_xc_sph_blue.append(interp_n_xc_sph_blue_gam)

    # sph exact results
    n_r0_sph_exact = np.load('n_r0_hookes_atom_sph_exact.npy')
    n_xc_sph_exact = blue_tools.n_CP_to_n_xc(n_r0_sph_exact, n)
    n_x_exact = blue_tools.get_two_el_n_x(n)

    interp_n_xc_sph_exact = blue_tools.get_interp_n_xc(grids, n_xc_sph_exact)
    interp_n_x_exact = blue_tools.get_interp_n_xc(grids, n_x_exact)

    # avg xc hole sph exact
    u_grids = np.linspace(0, 6, 200)
    avg_xc_hole_sph_exact = []
    avg_x_hole_exact = []
    for u in u_grids:
        avg_xc_hole_sph_exact.append(
            blue_tools.get_avg_xc_hole(u, grids, interp_n_xc_sph_exact, n))
        avg_x_hole_exact.append(
            blue_tools.get_avg_xc_hole(u, grids, interp_n_x_exact, n))

    avg_xc_hole_sph_exact = np.asarray(avg_xc_hole_sph_exact)
    avg_x_hole_exact = np.asarray(avg_x_hole_exact)
    avg_c_hole_sph_exact = (avg_xc_hole_sph_exact - avg_x_hole_exact)

    # avg xc hole sph blue

    avg_xc_hole_sph_blue = []
    for interp_n_xc_sph_blue_gam in interp_n_xc_sph_blue:
        avg_xc_hole_sph_blue_gam = []
        u_grids = np.linspace(0, 6, 200)
        for u in u_grids:
            avg_xc_hole_sph_blue_gam.append(
                blue_tools.get_avg_xc_hole(u, grids, interp_n_xc_sph_blue_gam,
                                           n))

        avg_xc_hole_sph_blue_gam = np.asarray(avg_xc_hole_sph_blue_gam)
        avg_xc_hole_sph_blue.append(avg_xc_hole_sph_blue_gam)

    for i, avg_xc_hole_sph_blue_gam in enumerate(avg_xc_hole_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(u_grids, avg_xc_hole_sph_blue_gam, label=plot_label)

        # print('U_xc sph blue = ',
        #      2 * np.trapz(2 * np.pi * u_grids * avg_xc_hole_sph_blue_gam, u_grids))

    plt.plot(u_grids, avg_xc_hole_sph_exact,
             label='sph. exact')

    plt.xlabel('$u$')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    for i, avg_xc_hole_sph_blue_gam in enumerate(avg_xc_hole_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(u_grids, 4 * np.pi * u_grids * avg_xc_hole_sph_blue_gam,
                 label=plot_label)

    plt.plot(u_grids, 4 * np.pi * u_grids * avg_xc_hole_sph_exact,
             label='sph. exact')

    plt.xlabel('$u$')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    for i, avg_xc_hole_sph_blue_gam in enumerate(avg_xc_hole_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(u_grids, (avg_xc_hole_sph_blue_gam - avg_x_hole_exact),
                 label=plot_label)

    plt.plot(u_grids, avg_c_hole_sph_exact,
             label='sph. exact')

    plt.xlabel('$u$')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    for i, avg_xc_hole_sph_blue_gam in enumerate(avg_xc_hole_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(u_grids, 4 * np.pi * u_grids * (
                avg_xc_hole_sph_blue_gam - avg_x_hole_exact),
                 label=plot_label)

    plt.plot(u_grids, 4 * np.pi * u_grids * avg_c_hole_sph_exact,
             label='sph. exact')

    plt.xlabel('$u$')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    sys.exit()

# get M(r) plots
if run == 'M(r)':
    grids = np.linspace(0.0056, 6, 1000)

    n = get_n(grids)

    # sph blue results
    n_xc_sph_blue = []
    M_xc_sph_blue = []
    for gam in gammas_files:
        n_r0_sph_blue_gam = np.load('n_r0_blue_gam_' + gam + '.npy')
        n_xc_sph_blue_gam = blue_tools.n_CP_to_n_xc(n_r0_sph_blue_gam, n)
        n_xc_sph_blue.append(n_xc_sph_blue_gam)
        M_xc_sph_blue_gam = blue_tools.radial_get_v_H_n_xc(grids,
                                                           n_xc_sph_blue_gam)
        M_xc_sph_blue.append(M_xc_sph_blue_gam)

    # sph exact results
    n_r0_sph_exact = np.load('n_r0_hookes_atom_sph_exact.npy')
    n_xc_sph_exact = blue_tools.n_CP_to_n_xc(n_r0_sph_exact, n)
    M_xc_sph_exact = blue_tools.radial_get_v_H_n_xc(grids, n_xc_sph_exact)
    n_x_exact = blue_tools.get_two_el_n_x(n)
    M_x_exact = blue_tools.radial_get_v_H_n_xc(grids, n_x_exact)

    # plot 0.5 n(r) M_xc(r). Integrating this * 4 pi r^2 = U_xc.
    for i, M_xc_sph_blue_gam in enumerate(M_xc_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(grids, 0.5 * n * M_xc_sph_blue_gam, label=plot_label)
    # plot sph exact
    plt.plot(grids, 0.5 * n * M_xc_sph_exact, label='sph. exact')
    # plot non-sph exact
    # look at mathematica. Use FortranForm.. integrate over theta and r'

    plt.xlabel('$r$')
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()

    # plot 0.5 * 4 pi r^2 n(r) M_xc(r). Integrating this = U_xc.
    for i, M_xc_sph_blue_gam in enumerate(M_xc_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(grids,
                 4 * np.pi * (grids * grids) * 0.5 * n * M_xc_sph_blue_gam,
                 label=plot_label)
    # plot sph exact
    plt.plot(grids, 4 * np.pi * (grids * grids) * 0.5 * n * M_xc_sph_exact,
             label='sph. exact')
    # plot non-sph exact
    # look at mathematica. Use FortranForm.. integrate over theta and r'

    plt.xlabel('$r$')
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()

    # plot 0.5 n(r) M_c(r). Integrating this * 4 pi r^2 = U_c.
    for i, M_xc_sph_blue_gam in enumerate(M_xc_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(grids, 0.5 * n * (M_xc_sph_blue_gam - M_x_exact),
                 label=plot_label)
    # plot sph exact
    plt.plot(grids, 0.5 * n * (M_xc_sph_exact - M_x_exact), label='sph. exact')
    # plot non-sph exact
    # look at mathematica. Use FortranForm.. integrate over theta and r'

    plt.xlabel('$r$')
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()

    # plot 0.5 * 4 pi r^2 n(r) M_c(r). Integrating this = U_c.
    for i, M_xc_sph_blue_gam in enumerate(M_xc_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(grids,
                 4 * np.pi * (grids * grids) * 0.5 * n * (
                         M_xc_sph_blue_gam - M_x_exact),
                 label=plot_label)
    # plot sph exact
    plt.plot(grids, 4 * np.pi * (grids * grids) * 0.5 * n * (
            M_xc_sph_exact - M_x_exact),
             label='sph. exact')
    # plot non-sph exact
    # look at mathematica. Use FortranForm.. integrate over theta and r'

    plt.xlabel('$r$')
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()

    sys.exit()
