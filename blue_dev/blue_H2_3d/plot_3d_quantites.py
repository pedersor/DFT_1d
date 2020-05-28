import matplotlib.pyplot as plt
import numpy as np
import sys


def get_plotting_params():
    # plotting parameters
    params = {'mathtext.default': 'default'}
    plt.rcParams.update(params)
    plt.rcParams['axes.axisbelow'] = True
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 9
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size
    fig, ax = plt.subplots()

    return fig, ax


def txt_file_to_array(file):
    # two column file to np arrays
    with open(file) as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]

    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def get_Vpp(R):
    return 1. / np.abs(R)


def check_Exc_paper():
    # Exc from paper
    R_idx_paper = [0, 2, 5, 10, 15]
    paper_Exc_vals = [-0.7857, -0.7024, -0.6174, -0.5543, -0.5642]

    for i, R_idx in enumerate(R_idx_paper):
        print(R[R_idx], '      ',
              paper_Exc_vals[i] - (E[R_idx] - (
                      2 * T_s[R_idx] + V_ext[R_idx] + 2 * U_plus_Ex[R_idx])))


def array_to_file(datafile_path, x_column, y_column, header):
    datafile_path += '.dat'
    data = np.array([x_column, y_column])
    data = data.T
    # here you transpose your data, so to have it in two columns

    with open(datafile_path, 'w+') as datafile_id:
        np.savetxt(datafile_id, data, fmt=['%d', '%d'], header=header,
                   comments='')

    return


def array_to_plot(datafile_path, x, y, xlabel, ylabel):
    return


def table_print(to_print, round_to_dec=3, last_in_row=False):
    rounded_to_print = format(to_print, '.' + str(round_to_dec) + 'f')
    if last_in_row:
        end = ' '
        print(rounded_to_print, end=end)
        print(r'\\')
        print('\hline')

    else:
        end = ' & '
        print(rounded_to_print, end=end)


if __name__ == '__main__':
    # exact results
    R, T = txt_file_to_array('H2_from_srwhite/T.dat')
    R, T_s = txt_file_to_array('H2_from_srwhite/T_s.dat')
    R, V_ext_plus_T = txt_file_to_array('H2_from_srwhite/H1en.dat')
    R, U_plus_Ex = txt_file_to_array('H2_from_srwhite/U.dat')
    R, E = txt_file_to_array('H2_from_srwhite/E.dat')
    R, Vee = txt_file_to_array('H2_from_srwhite/Vee.dat')
    V_ext = V_ext_plus_T - 2 * T

    E_xc = E - (2 * T_s + V_ext + 2 * U_plus_Ex)
    U_c = Vee - U_plus_Ex
    T_c = 2 * T - 2 * T_s
    E_c = E_xc - (-U_plus_Ex)
    U_xc = (-U_plus_Ex) + U_c
    E_x = -U_plus_Ex

    # half blue
    R, V_ee_blue_half = txt_file_to_array('H2_from_srwhite/Vee_blue_half.dat')
    U_c_blue_half = V_ee_blue_half - U_plus_Ex
    # gam results
    R, V_ee_blue_gam_2 = txt_file_to_array('H2_from_srwhite/Vee_blue_gam_2.dat')
    U_c_blue_gam_2 = V_ee_blue_gam_2 - U_plus_Ex

    R, V_ee_blue_gam_0_43 = txt_file_to_array('H2_from_srwhite/Vee_blue_gam_0_43.dat')
    U_c_blue_gam_0_43 = V_ee_blue_gam_0_43 - U_plus_Ex

    # blue results
    R, V_ee_blue = txt_file_to_array('H2_from_srwhite/Vee_blue.dat')
    U_c_blue = V_ee_blue - U_plus_Ex

    R, V_ee_blue = txt_file_to_array('H2_from_srwhite/Vee_blue_gam_0_43.dat')
    U_c_blue = V_ee_blue - U_plus_Ex

    # plots --------
    def do_plot():
        plt.xlabel('$R$', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(alpha=0.4)
        plt.show()


    # total energy dissociation
    plt.plot(R, 2 * T + V_ee_blue_half + V_ext + get_Vpp(R),
             label='$E_0[V^{Blue, e^B = 1/2}_{ee}(R)] + T^{Exact}_c[n(R)]$')
    plt.plot(R, 2 * T + V_ee_blue_gam_0_43 + V_ext + get_Vpp(R),
             label='$E_0[V^{Blue, \gamma = 0.43}_{ee}(R)] + T^{Exact}_c[n(R)]$')
    plt.plot(R, 2 * T + V_ee_blue_gam_2 + V_ext + get_Vpp(R),
             label='$E_0[V^{Blue, \gamma = 2}_{ee}(R)] + T^{Exact}_c[n(R)]$')
    plt.plot(R, 2 * T + V_ee_blue + V_ext + get_Vpp(R),
             label='$E_0[V^{Blue, e^B = 1}_{ee}(R)] + T^{Exact}_c[n(R)]$')
    plt.plot(R, E + get_Vpp(R),
             label='$E_0[V^{Exact}_{ee}(R)] + T^{Exact}_c[n(R)]$')

    do_plot()

    # U_c plot
    plt.plot(R, U_c_blue_half, label='$U^{Blue, e^B = 1/2}_c(R)$')
    plt.plot(R, U_c_blue_gam_0_43, label='$U^{Blue, \gamma = 0.43}_c(R)$')
    plt.plot(R, U_c_blue_gam_2, label='$U^{Blue, \gamma = 2}_c(R)$')
    plt.plot(R, U_c_blue, label='$U^{Blue, e^B = 1}_c(R)$')

    plt.plot(R, U_c, label='$U^{Exact}_c(R)$')

    do_plot()

    # new e/2 blue results from steve
    R_idx_steve = [0, 2, 5, 10, 15]

    for i, R_idx_val in enumerate(R_idx_steve):
        print(R[R_idx_val], end=" & ")
        table_print(E_x[R_idx_val])
        table_print(V_ee_blue[R_idx_val])
        table_print(Vee[R_idx_val])

        U_c_blue = V_ee_blue[R_idx_val] - U_plus_Ex[R_idx_val]
        U_xc_blue = (-U_plus_Ex[R_idx_val]) + U_c_blue

        table_print(U_xc_blue)
        table_print(U_xc[R_idx_val])

        table_print(U_c_blue)
        table_print(U_c[R_idx_val], last_in_row=True)

    sys.exit()
