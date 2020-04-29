import matplotlib.pyplot as plt
import numpy as np


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

if __name__ == '__main__':
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





    plt.xlabel('R')
    plt.legend()
    plt.show()
