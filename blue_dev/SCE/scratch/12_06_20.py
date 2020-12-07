import numpy as np
import functionals
import blue_tools
from scipy.interpolate import InterpolatedUnivariateSpline
import SCE.SCE_tools as SCE_tools


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


grids = np.arange(-256, 257) * 0.08

lam_lst = [1, 50, 100, 150, 200, 400]

for i, lam in enumerate(lam_lst):
    lam_dir = '../hookes_atom_linear/lam_' + str(lam)

    n_lam = np.load(lam_dir + '/densities.npy')[0]

    grids_interp, n_lam_interp = get_grids_n_interp(grids, n_lam)

    f_i_exact, Vee_SCE = get_Vee_SCE(grids_interp, n_lam_interp)



    """ blue quantities
    blue_CP_densities = np.load(
        'Hookes_atom_lam/blue_CP_densities_' + str(lam) + '.npy')

    blue_v_n_H = functionals.get_v_n_xc(grids, blue_CP_densities)
    blue_v_ee = 0.5 * np.trapz(blue_v_n_H * n_lam, grids)
    """

    """
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
    """
