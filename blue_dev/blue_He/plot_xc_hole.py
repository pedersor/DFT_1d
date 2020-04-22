import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import sys


def txt_file_to_array(file):
    # two column file to np arrays
    with open(file) as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]

    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def get_interp_n_xc(grids, n_xc):
    interp_n_xc = []
    for i, n_xc_r in enumerate(n_xc):
        # interpolated cubic spline function
        cubic_spline = InterpolatedUnivariateSpline(grids, n_xc_r, k=3)
        interp_n_xc.append(cubic_spline)

    interp_n_xc = np.asarray(interp_n_xc)
    return interp_n_xc


def get_avg_xc_hole(u, grids, interp_n_xc, n):
    # x values used in integral, see RP logbook 3/16/20
    x_grids = np.linspace(-1, 1, len(grids))
    n_xc_after_x_integral_r = []
    for i, r in enumerate(grids):
        interp_n_xc_r = interp_n_xc[i]
        u_plus_r = ((2 * u * r * x_grids) + (u * u) + (r * r)) ** (0.5)
        n_xc_after_x_integral = np.trapz(interp_n_xc_r(u_plus_r), x_grids)
        n_xc_after_x_integral_r.append(n_xc_after_x_integral)

    n_xc_after_x_integral_r = np.asarray(n_xc_after_x_integral_r)

    avg_xc_hole = np.pi * np.trapz(grids * grids * n * n_xc_after_x_integral_r,
                                   grids)

    return avg_xc_hole


if __name__ == '__main__':
    pi = np.pi
    L = 20
    grid_size = 1000
    grids = np.linspace(0.018435, L, grid_size)
    extrap_grids = np.linspace(0, L, grid_size)

    n_r0_r = np.load('n_r0.npy')
    n_HF = np.load('n_HF.npy')

    n_xc = []
    for n_r0 in n_r0_r:
        n_xc.append(-n_HF + n_r0)

    n_xc = np.asarray(n_xc)

    n_x = []
    for n_r0 in n_r0_r:
        n_x.append(-n_HF/2.)

    n_x = np.asarray(n_x)


    # interpolated n_xc and n_x
    interp_n_xc = get_interp_n_xc(grids, n_xc)
    interp_n_x = get_interp_n_xc(grids, n_x)

    # interoplated n
    interp_n_HF = InterpolatedUnivariateSpline(grids, n_HF, k=3)

    avg_xc_hole_ontop = get_avg_xc_hole(0, grids, interp_n_xc, n_HF)
    print('avg ontop hole = ', avg_xc_hole_ontop)

    u_grids = np.linspace(0, 3, 30)
    avg_xc_hole_u = []
    avg_x_hole_u = []
    for u in u_grids:
        avg_xc_hole_u.append(get_avg_xc_hole(u, grids, interp_n_xc, n_HF))
        avg_x_hole_u.append(get_avg_xc_hole(u, grids, interp_n_x, n_HF))

    avg_xc_hole_u = np.asarray(avg_xc_hole_u)
    avg_x_hole_u = np.asarray(avg_x_hole_u)

    plt.plot(u_grids, 2 * np.pi * u_grids * (avg_xc_hole_u - avg_x_hole_u),
             label=r'$2\pi u \left\langle n^B_{c}(u) \right\rangle$')



    # plot exact xc hole --------

    u_exact_grids, x_hole_exact = txt_file_to_array('Exact_x_Hole')

    u_exact_grids, xc_hole_exact = txt_file_to_array('Exact_xc_hole')

    plt.plot(u_exact_grids, xc_hole_exact - x_hole_exact,
             label=r'$2\pi u \left\langle n_{c}(u) \right\rangle$')

    plt.xlabel('u')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()
