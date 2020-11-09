import numpy as np
import matplotlib.pyplot as plt
import single_electron
import SCE.SCE_tools as SCE_tools
from scipy.interpolate import InterpolatedUnivariateSpline
import functionals
import blue_tools
import sys


def get_n_gam(grids, n, gam, large_grids):
    n_interp = InterpolatedUnivariateSpline(grids, n, k=3)
    grids_gam = gam * large_grids

    n_gam = gam * n_interp(grids_gam)
    return n_gam


def harmonic_osc_pot(grids, k=0.25):
    return 0.5 * k * (grids ** 2)


def exp_hydrogenic(grids, A=1.071295, k=(1. / 2.385345), Z=1):
    vp = Z * A * np.exp(-k * np.abs(grids))
    return vp


def get_n_CP_blue(large_grids, gam):
    v_ext = gam * gam * harmonic_osc_pot(gam * large_grids)

    blue_CP_densities = []
    for ref_x in large_grids:
        blue_v_ee = gam * gam * exp_hydrogenic(
            gam * (large_grids - ref_x))
        v_blue = blue_v_ee + v_ext

        def get_v_blue(large_grids):
            return v_blue

        solver = single_electron.EigenSolver(large_grids, get_v_blue)
        solver.solve_ground_state()
        blue_n_CP = solver.density

        blue_CP_densities.append(blue_n_CP)

    gam_str = dec_to_str(gam)
    np.save('Hookes_atom_gam/blue_CP_densities_' + gam_str + '.npy',
            blue_CP_densities)


def dec_to_str(dec):
    # e.g. 0.08 -> 0_08 filename convention

    dec_str = str(dec)
    dec_idx = dec_str.index('.')
    out_str = dec_str[:dec_idx] + '_' + dec_str[dec_idx + 1:]
    return out_str


example = 'blue_Vee'

if example == 'blue_Vee':
    gam = 0.005
    gam_str = dec_to_str(gam)

    grids = np.arange(-256, 257) * 0.08
    large_grids = np.arange(-256, 257) * 4

    n = np.load('Hookes_atom_gam/out_data_gam_1/densities.npy')[0]
    n_gam = get_n_gam(grids, n, gam, large_grids)

    plt.plot(large_grids, n_gam)
    plt.show()
    get_n_CP_blue(large_grids, gam)

    blue_CP_densities_gam = np.load(
        'Hookes_atom_gam/blue_CP_densities_' + str(gam_str) + '.npy')

    plt.plot(blue_CP_densities_gam[256])
    plt.show()

    blue_v_n_H = functionals.get_v_n_xc(large_grids, blue_CP_densities_gam)
    blue_v_ee = 0.5 * np.trapz(blue_v_n_H * n_gam, large_grids)

    print(blue_v_ee / gam)

    f_i_exact, v_ee_SCE_exact = SCE_tools.get_Vee_SCE(grids, n, N=2)

    print(v_ee_SCE_exact)
