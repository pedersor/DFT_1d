import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import blue_tools
import sys


def txt_file_to_array(file: object) -> object:
    # two column file to np arrays
    with open(file) as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]

    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


gammas_files = ['0', '0_43', '1', '1_5', '2', '3', 'inf']
gammas_disp = ['0', '0.43', r'1.5/\left\langle r_s \right\rangle \approx 1.0',
               '1.5', '2', '3', '\infty']

run = 'table'

L = 20
# for 1000, L = 20 using initial pt 0.018435 to minimize error of H and He
grid_size = 1000
grids = np.linspace(0.018435, L, grid_size)

n = np.load('n_HF.npy')

# get table results U_xc, U_c, etc.
if run == 'table':
    # sph blue results
    n_r0_sph_blue = []
    v_H_n_ee_blue = []
    V_ee_blue = []
    for gam in gammas_files:
        n_r0_sph_blue_gam = np.load('n_r0_Z_blue_gam_' + gam + '.npy')[0]
        n_r0_sph_blue.append(n_r0_sph_blue_gam)

        v_H_n_ee_blue_gam = blue_tools.radial_get_v_H_n_xc(grids,
                                                           n_r0_sph_blue_gam)
        v_H_n_ee_blue.append(v_H_n_ee_blue_gam)
        V_ee_blue_gam = blue_tools.radial_get_U_xc(grids, n, v_H_n_ee_blue_gam)
        V_ee_blue.append(V_ee_blue_gam)

    # HF results
    v_H_n = blue_tools.radial_get_v_H_n(grids, n)
    U_H = blue_tools.radial_get_U_xc(grids, n, v_H_n)
    E_x = -(U_H / 2)

    # non-sph exact results
    E_x_exact = -1.024568
    U_H_exact = -2 * E_x_exact

    V_ee_exact = 0.447443
    U_c_exact = -0.078750
    U_xc_exact = U_c_exact + E_x_exact

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

        # U_c non-sph exact error
        error_sph = 100 * (U_c_exact - U_c_blue_gam) / (U_c_exact)
        blue_tools.table_print(error_sph, round_to_dec=1, last_in_row=True)

    # non-sph exact method
    blue_tools.table_print('exact')
    blue_tools.table_print(U_xc_exact, round_to_dec=4)
    blue_tools.table_print(U_c_exact, round_to_dec=4)
    blue_tools.table_print(0., round_to_dec=1, last_in_row=True)

    sys.exit()

# get <n_xc(u)> plots
if run == 'n_xc':
    # sph blue results
    interp_n_xc_sph_blue = []
    for gam in gammas_files:
        n_r0_sph_blue_gam = np.load('n_r0_Z_blue_gam_' + gam + '.npy')[0]
        n_xc_sph_blue_gam = blue_tools.n_CP_to_n_xc(n_r0_sph_blue_gam, n)
        interp_n_xc_sph_blue_gam = blue_tools.get_interp_n_xc(grids,
                                                              n_xc_sph_blue_gam)
        interp_n_xc_sph_blue.append(interp_n_xc_sph_blue_gam)

    # n_x from HF density
    n_x_HF = blue_tools.get_two_el_n_x(n)
    interp_n_x_HF = blue_tools.get_interp_n_xc(grids, n_x_HF)

    u_grids = np.linspace(0, 3, 200)
    avg_x_hole_HF = []
    for u in u_grids:
        avg_x_hole_HF.append(
            blue_tools.get_avg_xc_hole(u, grids, interp_n_x_HF, n))

    avg_x_hole_HF = np.asarray(avg_x_hole_HF)

    # exact results
    u_exact_grids, x_hole_exact = txt_file_to_array('Exact_x_Hole')

    u_exact_grids, xc_hole_exact = txt_file_to_array('Exact_xc_hole')

    c_hole_exact = xc_hole_exact - x_hole_exact
    print('U_c exact = ', 2 * np.trapz(c_hole_exact, u_exact_grids))

    c_hole_exact = (xc_hole_exact - x_hole_exact)

    # avg xc hole sph blue
    avg_xc_hole_sph_blue = []
    for interp_n_xc_sph_blue_gam in interp_n_xc_sph_blue:
        avg_xc_hole_sph_blue_gam = []
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

    plt.plot(u_exact_grids, xc_hole_exact / (2 * np.pi * u_exact_grids),
             label='exact')

    plt.xlabel('$u$')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    for i, avg_xc_hole_sph_blue_gam in enumerate(avg_xc_hole_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(u_grids, 4 * np.pi * u_grids * avg_xc_hole_sph_blue_gam,
                 label=plot_label)

    plt.plot(u_exact_grids, 2 * xc_hole_exact,
             label='exact')

    plt.xlabel('$u$')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    for i, avg_xc_hole_sph_blue_gam in enumerate(avg_xc_hole_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(u_grids, (avg_xc_hole_sph_blue_gam - avg_x_hole_HF),
                 label=plot_label)

    plt.plot(u_exact_grids, c_hole_exact / (2 * np.pi * u_exact_grids),
             label='exact')

    plt.xlabel('$u$')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    for i, avg_xc_hole_sph_blue_gam in enumerate(avg_xc_hole_sph_blue):
        plot_label = 'blue ($\gamma = ' + gammas_disp[i] + '$)'
        plt.plot(u_grids, 4 * np.pi * u_grids * (
                avg_xc_hole_sph_blue_gam - avg_x_hole_HF),
                 label=plot_label)

    plt.plot(u_exact_grids, 2 * c_hole_exact,
             label='exact')

    plt.xlabel('$u$')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    sys.exit()

# get M(r) plots
# need exact here to compare... require P(r,r')..
if run == 'M(r)':
    # sph blue results
    n_xc_sph_blue = []
    M_xc_sph_blue = []
    for gam in gammas_files:
        n_r0_sph_blue_gam = np.load('n_r0_Z_blue_gam_' + gam + '.npy')
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
    plt.plot(grids, 0.5 * n * M_xc_sph_exact, label='exact')
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
             label='exact')
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
    plt.plot(grids, 0.5 * n * (M_xc_sph_exact - M_x_exact), label='exact')
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
             label='exact')
    # plot non-sph exact
    # look at mathematica. Use FortranForm.. integrate over theta and r'

    plt.xlabel('$r$')
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()

    sys.exit()
