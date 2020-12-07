import numpy as np
import matplotlib.pyplot as plt
import single_electron
import SCE.SCE_tools as SCE_tools
from scipy.interpolate import InterpolatedUnivariateSpline
import functionals
import blue_tools
import sys
from scipy.special import lambertw, erf, erfc
from scipy.optimize import leastsq, fsolve, bisect


def get_cumulant_funct_1d(grids, n_interpolant, x):
    """cumulant function for 1D density: N_e(x) = \int_{-\infty}^x dx' n(x')"""

    N_e = n_interpolant.integral(grids[0], x)

    return N_e


def get_N_e_inv_arg(split_grids, N_e, i=1, N=2):
    # split_grids: grids split between >/< args.

    less_than_arg = np.asarray([N_e(x) for x in split_grids[0]]) + i

    greater_than_arg = np.asarray([N_e(x) for x in split_grids[1]]) + i - N

    return less_than_arg, greater_than_arg


def get_N_e_inv(x, N_e):
    root_to_solve = lambda y: N_e(y) - x

    N_e_inv = bisect(root_to_solve, 0, 20)

    print(N_e_inv)
    return N_e_inv


lam = 400

# exact density
n = np.load('Hookes_atom_lam/out_data_' + str(lam) + '/densities.npy')[0]
# corresponding 1d grid used
grids = np.arange(-256, 257) * 0.08
grids_large = np.arange(-2560, 2561) * 0.008

grids_split_zero = np.split(grids, np.where(grids == 0)[0])
# delete zero from right-side in split
grids_split_zero[1] = grids_split_zero[1][1:]

grids_large_split_zero = np.split(grids_large, np.where(grids_large == 0)[0])
# delete zero from right-side in split
grids_large_split_zero[1] = grids_large_split_zero[1][1:]

n_interpolant = InterpolatedUnivariateSpline(grids, n, k=3)
N_e = lambda x: get_cumulant_funct_1d(grids, n_interpolant, x)

bias_slope = (1 - N_e(0.0)) / np.abs(grids[0])
bias = (grids_large - grids_large[0]) * bias_slope

N_e_interp = np.asarray([N_e(x) for x in grids_large])
N_e_interp += bias

'''
mask = np.diff(N_e_interp) > 0
N_e_interp = N_e_interp[1:]
plt.plot(mask)
plt.show()
'''

N_e_inv_interp = InterpolatedUnivariateSpline(N_e_interp[:2000],
                                              grids_large[:2000], k=3)

N_e_inv_arg = get_N_e_inv_arg(grids_large_split_zero, N_e)

# plt.plot(grids_split_zero[1][:80], N_e_inv_arg[1][:80])

plt.plot(grids_large_split_zero[1][:800], N_e_inv_interp(N_e_inv_arg[1][:800]))
plt.show()

sys.exit()
N_e_inv_arg_0, N_e_inv_arg_1 = get_N_e_inv_arg(grids_split_zero, N_e)

plt_0 = plt.plot(grids_split_zero[0], N_e_inv_arg_0)
plt.plot(grids_split_zero[1], N_e_inv_arg_1, color=plt_0[0].get_color())

plt.show()

plt.show()
