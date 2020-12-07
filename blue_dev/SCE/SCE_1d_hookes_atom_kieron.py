import numpy as np
import matplotlib.pyplot as plt
import single_electron
import SCE.SCE_tools as SCE_tools
from scipy.interpolate import InterpolatedUnivariateSpline
import functionals
import blue_tools
import sys
from scipy.special import lambertw, erf, erfc
from scipy.optimize import leastsq, fsolve, fmin
from scipy.integrate import quad, dblquad

from pynverse import inversefunc


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


def get_exact_cp_density(file):
    cp_densities = []
    cp_density = []
    for line in open(file):
        line = line.split('=')
        grid = float(line[0].split('_')[-1])
        if grid == 1:
            # start new cp density
            if cp_density:
                cp_densities.append(np.asarray(cp_density))
            cp_density = []

        cp_density.append(float(line[1].rstrip()) / 0.08 / 0.08)

    # append last cp density
    cp_densities.append(np.asarray(cp_density))

    cp_densities = np.asarray(cp_densities)
    return cp_densities


def get_gauss_Psi(x1, x2, c, d):
    coeff = np.sqrt(c / (1 + np.exp(-4 * c * d * d)) / np.pi)
    term1 = np.exp(-c * (d + x1) ** 2 - c * (-d + x2) ** 2)
    term2 = np.exp(-c * (-d + x1) ** 2 - c * (d + x2) ** 2)

    return coeff * (term1 + term2)


example = 'comotion_ftns_all_in_one'

if example == 'ref_point_example':
    # 1D Hooke's atom reference point example
    h = 0.08
    grids = np.arange(-256, 257) * h
    v_ext = harmonic_osc_pot(grids)

    # reference point (x)
    ref_x = 5.04
    ref_x_idx = int(np.where(grids == ref_x)[0][0])

    # to plot: 'density', 'potential'
    plot = 'density'

    lambda_lst = [100, 200, 400]
    for lam in lambda_lst:
        # exact results
        dir = 'Hookes_atom_lam/out_data_' + str(lam) + '/'
        n = np.load(dir + 'densities.npy')[0]

        plt.plot(grids, n)
        plt.show()

        d = grids[np.argmax(n)]
        c = n[np.argmax(n)]

        print(c, d)

        sys.exit()

        file = dir + 'cp_density.txt'
        exact_cp_density = get_exact_cp_density(file)[ref_x_idx]
        exact_cp_density /= n[ref_x_idx]

        # SCE results
        grids_interp, n_lam_interp = get_grids_n_interp(grids, n)
        f_i_exact, Vee_SCE = get_Vee_SCE(grids_interp, n_lam_interp)
        grids_interp_ref_x_idx = int(np.where(grids_interp == ref_x)[0][0])
        f_i_exact_ref_x = f_i_exact[0][grids_interp_ref_x_idx]

        # blue approximation
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
            # plot exact CP density
            blue_plot = plt.plot(grids, exact_cp_density,
                                 label='$\lambda = ' + str(lam) + '$')
            # plot blue approx. CP density
            plt.plot(grids, blue_n_CP, linestyle='dashed',
                     color=
                     blue_plot[0].get_color())

            # plot exact SCE CP density
            plt.vlines(f_i_exact_ref_x, 0, 1, color=blue_plot[0].get_color(),
                       alpha=0.6)

    plt.xlabel('$x^{\prime}$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

elif example == 'analytic_comotion_fit':
    # doesn't seem to work well... inverse does not exist for N_e analytical.

    N = 2  # num of electrons

    lam = 400

    f_i_exact = []

    # exact density
    n = np.load('Hookes_atom_lam/out_data_' + str(lam) + '/densities.npy')[0]
    # corresponding 1d grid used
    grids = np.arange(-256, 257) * 0.08


    def double_gaussian(x, params):
        (ampl, center, width) = params
        res = ampl * np.exp(- width * (x - center) ** 2.0) \
              + ampl * np.exp(-width * (x + center) ** 2.0)
        return res


    def double_gaussian_fit(params):
        fit = double_gaussian(grids, params)
        return (fit - n) ** 2


    def N_e_gauss(x, fit_params, value=0):
        (a, c, b) = fit_params
        factor = a * np.sqrt(np.pi) / (2 * np.sqrt(b))
        parens = 1 + erf(np.sqrt(b) * (x + c)) + erfc(np.sqrt(b) * (c - x))
        return factor * parens - value


    def inv_N_e(value, fit_params):

        res = fsolve(N_e_gauss, value, (fit_params, value))

        return res


    guess_params = [0.5, 5, 0.8]
    fit = leastsq(double_gaussian_fit, guess_params)
    fit_params = fit[0]

    N_e_exact = SCE_tools.get_cumulant_funct_1d(grids, n)

    N_e_gauss_fn = lambda x: N_e_gauss(x, fit_params)

    N_e_gauss_fn_inv = inversefunc(N_e_gauss_fn, y_values=grids)

    plt.plot(grids, N_e_gauss_fn(grids), label='gauss $N_e(x)$')
    plt.plot(grids, N_e_exact)
    plt.plot(grids, N_e_gauss_fn_inv)

    plt.xlabel('x', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

elif example == 'comotion_ftns':
    N = 2  # num of electrons

    lam = 400

    f_i_exact = []

    # exact density
    n = np.load('Hookes_atom_lam/out_data_' + str(lam) + '/densities.npy')[0]
    # corresponding 1d grid used
    grids = np.arange(-256, 257) * 0.08

    # blue CP density
    blue_cp_densities = np.load(
        'Hookes_atom_lam/blue_CP_densities_' + str(lam) + '.npy')

    # simple interpolation gives finer resolution of co-motion ftns.
    n = InterpolatedUnivariateSpline(grids, n, k=3)
    # evaluate on denser grid
    grids_interp = np.arange(-2560, 2561) * 0.008
    n = n(grids_interp)

    for i in np.arange(1, N):
        f_i = SCE_tools.get_f_i_1d(grids_interp, i, n, N)
        f_i_exact.append(f_i)
    f_i_exact = np.asarray(f_i_exact)

    f_i_blue = []
    for blue_n_cp in blue_cp_densities:
        n_cp_blue_interp = InterpolatedUnivariateSpline(grids, blue_n_cp, k=3)
        n_cp_blue_interp = n_cp_blue_interp(grids_interp)

        n_cp_blue_peak_location = grids_interp[np.argmax(n_cp_blue_interp)]
        f_i_blue.append(n_cp_blue_peak_location)
    f_i_blue = np.asarray(f_i_blue)

    # use scatter since divergence at origin is messy..
    plt.plot(grids, f_i_blue,
             label='$\lambda = ' + str(lam) + '$')
    plt.plot(grids_interp, f_i_exact[0])

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$f(x)$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

elif example == 'comotion_ftns_all_in_one':
    N = 2  # num of electrons

    lambda_lst = [100, 200, 400]
    alpha_lst = [0.3, 0.6, 1.]

    lambda_lst = [100]
    alpha_lst = [1.]

    for j, lam in enumerate(lambda_lst):
        f_i_exact = []

        # exact density
        n = np.load('Hookes_atom_lam/out_data_' + str(lam) + '/densities.npy')[
            0]
        # corresponding 1d grid used
        grids = np.arange(-256, 257) * 0.08

        # gaussian exact mimic. HL-wavefunction
        c = 0.423
        d = 4.48
        Psi_sqrd = lambda x1, x2: get_gauss_Psi(x1, x2, c=c, d=d) ** 2

        """ 
        # gauss density
        density = lambda x: 2 * quad(Psi_sqrd, -20, 20, x)[0]
        density_grid = np.asarray([density(x) for x in grids])
        """

        f_i_gauss = []
        for grid in grids:
            gauss_cp_density = lambda x2: -Psi_sqrd(grid, x2)
            x_min_gauss = fmin(gauss_cp_density, grid)

            f_i_gauss.append(x_min_gauss[0])

        f_i_gauss = np.asarray(f_i_gauss)

        # blue CP density
        blue_cp_densities = np.load(
            'Hookes_atom_lam/blue_CP_densities_' + str(lam) + '.npy')

        # exact CP density
        dir = 'Hookes_atom_lam/out_data_' + str(lam) + '/'
        file = dir + 'cp_density.txt'
        exact_cp_densities = get_exact_cp_density(file)
        exact_cp_densities = np.asarray(
            [n_cp / n_ref_x for n_cp, n_ref_x in zip(exact_cp_densities, n)])
        f_i_dmrg = []
        for exact_n_cp in exact_cp_densities:
            n_cp_exact_peak_location = grids[np.argmax(exact_n_cp)]
            f_i_dmrg.append(n_cp_exact_peak_location)
        f_i_dmrg = np.asarray(f_i_dmrg)

        # simple interpolation gives finer resolution of co-motion ftns.
        n = InterpolatedUnivariateSpline(grids, n, k=3)
        # evaluate on denser grid
        grids_interp = np.arange(-2560, 2561) * 0.008
        n = n(grids_interp)

        for i in np.arange(1, N):
            f_i = SCE_tools.get_f_i_1d(grids_interp, i, n, N)
            f_i_exact.append(f_i)
        f_i_exact = np.asarray(f_i_exact)

        f_i_blue = []
        for blue_n_cp in blue_cp_densities:
            n_cp_blue_interp = InterpolatedUnivariateSpline(grids, blue_n_cp,
                                                            k=3)
            n_cp_blue_interp = n_cp_blue_interp(grids_interp)

            n_cp_blue_peak_location = grids_interp[np.argmax(n_cp_blue_interp)]
            f_i_blue.append(n_cp_blue_peak_location)
        f_i_blue = np.asarray(f_i_blue)

        plt.plot(grids, f_i_blue, color='C0', alpha=alpha_lst[j], label='blue')

        # plt.plot(grids_interp, f_i_exact[0], color='C1', alpha=alpha_lst[j])
        plt.plot(grids, f_i_dmrg, color='C2', alpha=alpha_lst[j], label='dmrg')
        plt.plot(grids, f_i_gauss, color='C3', alpha=alpha_lst[j],
                 label='gauss HL')

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$f(x)$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

elif example == 'get n_CP':

    lam = 400

    grids_interp = np.arange(-256, 257) * 0.08
    v_ext = harmonic_osc_pot(grids_interp)

    blue_CP_densities = []
    for ref_x in grids_interp:
        blue_v_ee = lam * exp_hydrogenic(grids_interp - ref_x)
        v_blue = blue_v_ee + v_ext


        def get_v_blue(grids):
            return v_blue


        solver = single_electron.EigenSolver(grids_interp, get_v_blue)
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

        blue_CP_densities = np.load(
            'Hookes_atom_lam/blue_CP_densities_' + str(lam) + '.npy')

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

        blue_tools.table_print(lam, round_to_dec=0)
        blue_tools.table_print(lam * blue_v_ee)
        blue_tools.table_print(exact_Vee[i])
        blue_tools.table_print(lam * Vee_SCE)

        per_error_blue_dmrg = 100 * (exact_Vee[i] - lam * blue_v_ee) / (
            exact_Vee[i])
        blue_tools.table_print(per_error_blue_dmrg)

        per_error_blue_SCE = 100 * (lam * Vee_SCE - lam * blue_v_ee) / (
                lam * Vee_SCE)
        blue_tools.table_print(per_error_blue_SCE, last_in_row=True)

elif example == 'nuclear':
    grids_interp = np.arange(-256, 257) * 0.08

    lam = 100
    lam_dir = 'Hookes_atom_lam/out_data_' + str(lam)
    n_lam = np.load(lam_dir + '/densities.npy')[0]

    blue_CP_densities = np.load('blue_CP_densities_' + str(lam) + '.npy')

    blue_CP_densities_T = blue_CP_densities.T

    blue_CP_gradients = []
    for i, n_CP in enumerate(blue_CP_densities_T):
        phi_CP = np.sqrt(n_CP)
        phi_CP_grad = np.gradient(phi_CP, grids_interp)
        phi_CP_grad_grad = np.gradient(phi_CP_grad, grids_interp)

        blue_CP_gradients.append(phi_CP_grad_grad)

    blue_CP_gradients = np.asarray(blue_CP_gradients)
    blue_CP_gradients = blue_CP_gradients.T

    blue_CP_grad_grad_pot = []
    for i, blue_CP_grad in enumerate(blue_CP_gradients):
        blue_CP_grad_grad_pot.append(
            -0.5 * blue_CP_grad / np.sqrt(blue_CP_densities[i]))

    blue_CP_grad_grad_pot = np.asarray(blue_CP_grad_grad_pot)

    plt.plot(grids_interp, blue_CP_grad_grad_pot[256])
    plt.plot(grids_interp, blue_CP_grad_grad_pot[280])
    plt.plot(grids_interp, blue_CP_grad_grad_pot[300])

    plt.ylim(-100, 100)
    plt.show()

elif example == 'gaussians':

    grids_interp = np.arange(-256, 257) * 0.08

    lam_lst = [1, 50, 100, 150, 200, 400]

    for i, lam in enumerate(lam_lst):
        lam_dir = 'Hookes_atom_lam/out_data_' + str(lam)

        n_lam = np.load(lam_dir + '/densities.npy')[0]

        blue_CP_densities = np.load(
            'Hookes_atom_lam/blue_CP_densities_' + str(lam) + '.npy')

        gauss_CP_densities = np.asarray(
            [getGaussianCP(grids_interp, ref_x, lam) for ref_x in grids_interp])

        blue_v_n_H = functionals.get_v_n_xc(grids_interp, blue_CP_densities)
        blue_v_ee = 0.5 * np.trapz(blue_v_n_H * n_lam, grids_interp)

        blue_gauss_v_n_H = functionals.get_v_n_xc(grids_interp,
                                                  gauss_CP_densities)
        blue_gauss_v_ee = 0.5 * np.trapz(blue_gauss_v_n_H * n_lam, grids_interp)

        plt.plot(grids_interp, blue_v_n_H)
        plt.plot(grids_interp, blue_gauss_v_n_H)
        plt.show()

        blue_tools.table_print(lam, round_to_dec=0)
        blue_tools.table_print(lam * blue_v_ee)
        blue_tools.table_print(lam * blue_gauss_v_ee)

        per_error_blue_gauss = 100 * (
                lam * blue_v_ee - lam * blue_gauss_v_ee) / (
                                       lam * blue_v_ee)
        blue_tools.table_print(per_error_blue_gauss, last_in_row=True)
