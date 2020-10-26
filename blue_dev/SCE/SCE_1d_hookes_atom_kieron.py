import numpy as np
import matplotlib.pyplot as plt
import single_electron
import SCE.SCE_tools as SCE_tools
from scipy.interpolate import InterpolatedUnivariateSpline
import functionals
import blue_tools
import sys
from scipy.special import lambertw


def exp_hydrogenic(grids, A=1.071295, k=(1. / 2.385345), Z=1):
    vp = Z * A * np.exp(-k * np.abs(grids))
    return vp


def harmonic_osc_pot(grids, k=0.25):
    return 0.5 * k * (grids ** 2)


def getGaussianCP(grids, x_ref, lam):
    factor = 1
    if x_ref > 0:
        x_ref = -1 * x_ref
        factor = -1

    location = 2.38535 * lambertw(0.753124 * np.exp(0.419227 * x_ref) * lam)
    location = factor * np.real(location) + factor * 0.1

    k_constant = 0.25 + 0.188281 * np.exp((0.419227 * x_ref) - (
        lambertw(0.753124 * np.exp(0.419227 * x_ref) * lam))) * lam
    k_constant = np.real(k_constant)

    gaussian_n_CP = np.exp(-np.sqrt(k_constant) * ((grids - location) ** 2))
    norm = 1 / (np.trapz(gaussian_n_CP, grids))
    gaussian_n_CP = norm * gaussian_n_CP

    return gaussian_n_CP


def get_grids_n_interp(grids, n):
    # simple interpolation gives finer resolution of co-motion ftns.
    n_interp = InterpolatedUnivariateSpline(grids, n, k=3)
    # evaluate on denser grid
    grids_interp = np.arange(-2560, 2561) * 0.008

    n_interp = n_interp(grids_interp)

    return grids_interp, n_interp


def get_Vee_SCE(grids, n, N=2):
    f_i_exact = []

    for i in np.arange(1, N):
        f_i = SCE_tools.get_f_i_1d(grids, i, n, N)
        f_i_exact.append(f_i)

    f_i_exact = np.asarray(f_i_exact)

    v_ee_SCE_exact = SCE_tools.get_v_ee_SCE_1d(grids, n,
                                               f_i_exact)
    return f_i_exact, v_ee_SCE_exact


def get_blue_comotion_ft(grids, blue_CP_densities):
    blue_comotion_ft = []
    for blue_n_CP in blue_CP_densities:
        min_x_idx = np.argmax(blue_n_CP)
        blue_comotion_ft.append(grids[min_x_idx])

    blue_comotion_ft = np.asarray(blue_comotion_ft)
    return blue_comotion_ft


example = 'gaussians'

if example == 'ref_point_example':
    # 1D Hooke's atom reference point example

    grids = np.arange(-256, 257) * 0.08
    v_ext = harmonic_osc_pot(grids)

    # reference point (x)
    ref_x = 0.0

    # to plot: 'density', 'potential'
    plot = 'density'

    lambda_lst = [400]
    for lam in lambda_lst:
        blue_v_ee = lam * exp_hydrogenic(grids - ref_x)
        v_blue = blue_v_ee + v_ext


        def get_v_blue(grids):
            return v_blue


        solver = single_electron.EigenSolver(grids, get_v_blue)
        solver.solve_ground_state()
        blue_n_CP = solver.density

        gaussian_n_CP = getGaussianCP(grids, ref_x, lam)

        if plot == 'potential':
            # plot blue potential
            plt.plot(grids, v_blue, label='$\lambda = ' + str(lam) + '$')
            plt.ylabel('$\lambda v_{ee}(x - x^{\prime}) + v_{ext}(x^{\prime})$',
                       fontsize=16)
        elif plot == 'density':
            # plot blue CP density
            plt.plot(grids, blue_n_CP, label='$\lambda = ' + str(lam) + '$')
            # plot gaussian approx.
            plt.plot(grids, gaussian_n_CP, label='$\lambda = ' + str(lam) + '$')
            plt.ylabel('Blue CP density', fontsize=16)

    plt.xlabel('$x^{\prime}$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

elif example == 'comotion_ftns':
    # 1D Hooke's atom co-motion functions example

    grids = np.arange(-256, 257) * 0.08
    v_ext = harmonic_osc_pot(grids)

    f_i_exact = []
    N = 2  # num of electrons
    # exact density
    n = np.load('Hookes_atom/densities.npy')[0]
    # corresponding 1d grid used
    grids = np.arange(-256, 257) * 0.08

    # simple interpolation gives finer resolution of co-motion ftns.
    n = InterpolatedUnivariateSpline(grids, n, k=3)
    # evaluate on denser grid
    grids = np.arange(-2560, 2561) * 0.008
    n = n(grids)

    for i in np.arange(1, N):
        f_i = SCE_tools.get_f_i_1d(grids, i, n, N)
        f_i_exact.append(f_i)

    f_i_exact = np.asarray(f_i_exact)

    v_ee_SCE_exact = SCE_tools.get_v_ee_SCE_1d(grids, n,
                                               f_i_exact)
    print('v_ee_SCE_exact: ', v_ee_SCE_exact)

    lambda_lst = [1, 2, 5, 10, 20, 50, 100]

    for lam in lambda_lst:
        f_i_blue = []
        for r_idx, grid in enumerate(grids):
            v_SCE_blue = np.zeros(len(grids))
            v_SCE_blue += v_ext + lam * exp_hydrogenic(
                grids - grid)
            v_SCE_blue_min_idx = np.argmin(v_SCE_blue)
            f_i_blue.append(grids[v_SCE_blue_min_idx])

        f_i_blue = np.asarray(f_i_blue)

        plt.scatter(grids, f_i_blue, label='$\lambda = ' + str(lam) + '$',
                    marker='.')

    # use scatter since divergence at origin is messy..
    plt.scatter(grids, f_i_exact[0], marker='.', label='exact $f(x)$')

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$f(x)$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

elif example == 'dmrg':
    lam_lst = [50, 100, 150]

    grids = np.arange(-256, 257) * 0.08

    n_50 = np.load('Hookes_atom_lam/out_data_50/densities.npy')[0]
    n_100 = np.load('Hookes_atom_lam/out_data_100/densities.npy')[0]
    n_150 = np.load('Hookes_atom_lam/out_data_150_old/densities.npy')[0]
    n_lst = [n_50, n_100, n_150]

    for i, n in enumerate(n_lst):
        plt.plot(grids, n, label='$\lambda = ' + str(lam_lst[i]) + '$')

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('density', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

elif example == 'get n_CP':

    lam = 400

    grids = np.arange(-256, 257) * 0.08
    v_ext = harmonic_osc_pot(grids)

    blue_CP_densities = []
    for ref_x in grids:
        blue_v_ee = lam * exp_hydrogenic(grids - ref_x)
        v_blue = blue_v_ee + v_ext


        def get_v_blue(grids):
            return v_blue


        solver = single_electron.EigenSolver(grids, get_v_blue)
        solver.solve_ground_state()
        blue_n_CP = solver.density

        blue_CP_densities.append(blue_n_CP)

    np.save('blue_CP_densities_400.npy', blue_CP_densities)

elif example == 'calculate V_ee':

    grids = np.arange(-256, 257) * 0.08

    lam_lst = [1, 50, 100, 150, 200, 400]
    exact_Vee = [0.6034417, 2.299617, 2.69303, 2.92154, 3.0859, 3.488728421]

    for i, lam in enumerate(lam_lst):
        lam_dir = 'Hookes_atom_lam/out_data_' + str(lam)

        n_lam = np.load(lam_dir + '/densities.npy')[0]

        grids_interp, n_lam_interp = get_grids_n_interp(grids, n_lam)

        f_i_exact, Vee_SCE = get_Vee_SCE(grids_interp, n_lam_interp)

        blue_CP_densities = np.load('blue_CP_densities_' + str(lam) + '.npy')

        # extras
        '''
        blue_f_i = [get_blue_comotion_ft(grids, blue_CP_densities)]
        
        print(lam * Vee_SCE)
        blue_V_SCE = SCE_tools.get_v_ee_SCE_1d(grids, n_lam, blue_f_i)
        print(lam * blue_V_SCE)
        
        plt.plot(grids_interp, n_lam_interp)
        plt.plot(grids, blue_f_i[0])
        plt.plot(grids_interp, f_i_exact[0])
        plt.show()
        '''

        blue_v_n_H = functionals.get_v_n_xc(grids, blue_CP_densities)
        blue_v_ee = 0.5 * np.trapz(blue_v_n_H * n_lam, grids)

        blue_tools.table_print(lam * blue_v_ee)
        blue_tools.table_print(exact_Vee[i])
        blue_tools.table_print(lam * Vee_SCE)

        per_error_blue_dmrg = 100 * (exact_Vee[i] - lam * blue_v_ee) / (
            exact_Vee[i])

        per_error_blue_dmrg = (exact_Vee[i] - lam * blue_v_ee)
        blue_tools.table_print(per_error_blue_dmrg)

        per_error_blue_SCE = 100 * (lam * Vee_SCE - lam * blue_v_ee) / (
                lam * Vee_SCE)
        per_error_blue_SCE = (lam * Vee_SCE - lam * blue_v_ee)
        blue_tools.table_print(per_error_blue_SCE, last_in_row=True)

elif example == 'nuclear':
    grids = np.arange(-256, 257) * 0.08

    lam = 100
    lam_dir = 'Hookes_atom_lam/out_data_' + str(lam)
    n_lam = np.load(lam_dir + '/densities.npy')[0]

    blue_CP_densities = np.load('blue_CP_densities_' + str(lam) + '.npy')

    blue_CP_densities_T = blue_CP_densities.T

    blue_CP_gradients = []
    for i, n_CP in enumerate(blue_CP_densities_T):
        phi_CP = np.sqrt(n_CP)
        phi_CP_grad = np.gradient(phi_CP, grids)
        phi_CP_grad_grad = np.gradient(phi_CP_grad, grids)

        blue_CP_gradients.append(phi_CP_grad_grad)

    blue_CP_gradients = np.asarray(blue_CP_gradients)
    blue_CP_gradients = blue_CP_gradients.T

    blue_CP_grad_grad_pot = []
    for i, blue_CP_grad in enumerate(blue_CP_gradients):
        blue_CP_grad_grad_pot.append(
            -0.5 * blue_CP_grad / np.sqrt(blue_CP_densities[i]))

    blue_CP_grad_grad_pot = np.asarray(blue_CP_grad_grad_pot)

    plt.plot(grids, blue_CP_grad_grad_pot[256])
    plt.plot(grids, blue_CP_grad_grad_pot[280])
    plt.plot(grids, blue_CP_grad_grad_pot[300])

    plt.ylim(-100, 100)
    plt.show()

elif example == 'gaussians':

    grids = np.arange(-256, 257) * 0.08

    lam_lst = [1, 50, 100, 150, 200, 400]

    for i, lam in enumerate(lam_lst):
        lam_dir = 'Hookes_atom_lam/out_data_' + str(lam)

        n_lam = np.load(lam_dir + '/densities.npy')[0]

        blue_CP_densities = np.load('blue_CP_densities_' + str(lam) + '.npy')

        gauss_CP_densities = np.asarray(
            [getGaussianCP(grids, ref_x, lam) for ref_x in grids])

        blue_v_n_H = functionals.get_v_n_xc(grids, blue_CP_densities)
        blue_v_ee = 0.5 * np.trapz(blue_v_n_H * n_lam, grids)

        blue_gauss_v_n_H = functionals.get_v_n_xc(grids, gauss_CP_densities)
        blue_gauss_v_ee = 0.5 * np.trapz(blue_gauss_v_n_H * n_lam, grids)

        plt.plot(grids, blue_v_n_H)
        plt.plot(grids, blue_gauss_v_n_H)
        plt.show()

        blue_tools.table_print(lam, round_to_dec=0)
        blue_tools.table_print(lam * blue_v_ee)
        blue_tools.table_print(lam * blue_gauss_v_ee)

        per_error_blue_gauss = 100 * (
                lam * blue_v_ee - lam * blue_gauss_v_ee) / (
                                       lam * blue_v_ee)
        blue_tools.table_print(per_error_blue_gauss, last_in_row=True)
