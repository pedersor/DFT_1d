import matplotlib.pyplot as plt
import numpy as np
import sys
import blue_tools


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


run = 'disp'

# avg r_s vs R
if run == 'r_s':
    R, avg_r_s = blue_tools.txt_file_to_array('H2_from_srwhite/avg_r_s.dat',
                                              header=True)

    for i, avg_r_s_R in enumerate(avg_r_s):
        blue_tools.table_print(R[i], round_to_dec=1)
        blue_tools.table_print(avg_r_s_R, round_to_dec=4, last_in_row=True)

    sys.exit()

# dispersion plots etc.
if run == 'disp':
    # exact results

    _, T = txt_file_to_array('H2_from_srwhite/T.dat')
    # remove R > 4 values
    T = T[:-4]
    _, T_s = txt_file_to_array('H2_from_srwhite/T_s.dat')
    T_s = T_s[:-4]
    _, U_plus_Ex = txt_file_to_array('H2_from_srwhite/U_plus_Ex.dat')
    U_plus_Ex = U_plus_Ex[:-4]
    _, E = txt_file_to_array('H2_from_srwhite/E.dat')
    E = E[:-4]
    _, Vee = txt_file_to_array('H2_from_srwhite/Vee.dat')
    Vee = Vee[:-4]
    R, V_ext_plus_T = txt_file_to_array('H2_from_srwhite/H1en.dat')

    V_ext = V_ext_plus_T - 2 * T

    E_xc = E - (2 * T_s + V_ext + 2 * U_plus_Ex)
    U_c = Vee - U_plus_Ex
    T_c = 2 * T - 2 * T_s
    E_c = E_xc - (-U_plus_Ex)
    U_xc = (-U_plus_Ex) + U_c
    E_x = -U_plus_Ex

    gam_files = ['3o2r_s', '1or_s', '0', '0_43', '1_5', 'inf']
    gam_disp = ['3/(2r_s)', '1/r_s', '0', '0.43', '1.5', r'\infty']

    gam_files = ['1or_s', 'inf', '0']
    gam_disp = ['1/r_s(n(r))', r'\infty', '0']

    V_ee_blue = []
    U_c_blue = []
    for gam in gam_files:
        _, V_ee_blue_gam = txt_file_to_array(
            'H2_from_srwhite/Vee_blue_gam_' + gam + '.dat')

        if gam == '1or_s':
            # remove R > 4 values
            V_ee_blue_gam = V_ee_blue_gam[:-4]

        U_c_blue_gam = V_ee_blue_gam - U_plus_Ex

        V_ee_blue.append(V_ee_blue_gam)
        U_c_blue.append(U_c_blue_gam)


    # plots --------
    def do_plot():
        plt.xlabel('$R$', fontsize=18)
        plt.legend(fontsize=16)
        # plt.xlim(1,4)
        # plt.ylim(-.2,0)
        plt.grid(alpha=0.4)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()


    # energy of two infinitely separated H atoms
    two_H_atoms_E = -2 * 0.5

    # total energy dissociation
    for i, gam in enumerate(gam_disp):
        plt.plot(R, (2 * T + V_ee_blue[i] + V_ext + get_Vpp(R)) - two_H_atoms_E,
                 label='Blue')

    plt.plot(R, (E + get_Vpp(R)) - two_H_atoms_E, label='Exact')
    plt.ylabel('$E_0(R)$', fontsize=18)
    do_plot()

    # U_c plot
    out_column = [R]
    for i, gam in enumerate(gam_disp):
        out_column.append(U_c_blue[i])
        plt.plot(R, U_c_blue[i], label='$\gamma = ' + gam + '$', linewidth=3)

    out_column.append(U_c)
    plt.plot(R, U_c, label='Exact', color='black', linewidth=3)
    plt.ylabel('$U_c(R)$', fontsize=18)

    # write to multi-column file
    with open("blue_U_c_R.dat", "w") as file:
        file.write("R\t r_s(n(r)) \t r_s->0  \t r_s -> \inf \t exact \n")
        for x in zip(*out_column):
            R_print = x[0]
            to_print = [format(val, '.6f') for val in x[1:]]

            file.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(R_print, *to_print))

    do_plot()

    # table results for a blue approx.
    V_ee_blue = V_ee_blue[0]
    U_c_blue = V_ee_blue - U_plus_Ex

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
