import single_electron, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


# plotting parameters
params = {'mathtext.default': 'default'}
plt.rcParams.update(params)
plt.rcParams['axes.axisbelow'] = True
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
fig, ax = plt.subplots()

# suggested values for exponential hydrogenic systems,
# see Tom Baker et. al. PHYSICAL REVIEW B 91, 235141 (2015)
pi = np.pi
A = 1.071295
k_inv = 2.385345
k = 1. / k_inv

# continuous exponential well ------------------------------
num_electrons = 4
d_list = [5, 6, 7, 8, 10]

for d in d_list:
    root_list = ext_potentials.exp_hydro_cont_well_roots(A, k, d=d)
    energy_list = [-(1 / 8) * k ** 2 * x ** 2 for x in root_list]
    th_val = energy_list[0]  # theoretical/analytical value

    num_grids_list = [40, 80, 120, 160, 200, 400, 600, 800, 1000, 1200]
    num_grids_list = [x + 1 for x in num_grids_list]

    E_error = []
    for grid in num_grids_list:
        L = 25 + (d - 5) * 5
        if d == 5:
            L = 50

        grids = np.linspace(-L / 2, L / 2, grid)
        solver = single_electron.EigenSolver(grids, potential_fn=functools.partial(
            ext_potentials.exp_hydro_cont_well, A=A, k=k, d=d, a=0), end_points=True)
        solver.solve_ground_state()
        E_h2 = solver.eigenvalues[0]

        grids = np.linspace(-L / 2, L / 2, ((grid - 1) / 2) + 1)
        solver = single_electron.EigenSolver(grids, potential_fn=functools.partial(
            ext_potentials.exp_hydro_cont_well, A=A, k=k, d=d, a=0), end_points=True)
        solver.solve_ground_state()
        E_h = solver.eigenvalues[0]

        # Richardson Extrapolation
        E = ((4 * E_h2) - E_h) / 3.

        error = E - th_val
        E_error.append(np.abs(error))

    log_ngl = [np.log10(x) for x in num_grids_list]
    log_E = [np.log10(x) for x in E_error]

    # skip first 3 small N values for finding linear fit
    log_ngl_fit = log_ngl[3:]
    log_E_fit = log_E[3:]

    # linear fitting
    b, p = polyfit(log_ngl_fit, log_E_fit, 1)
    r2 = '%.4f' % (rsquared(log_ngl_fit, log_E_fit))
    yfit = [10 ** (b + p * xi) for xi in log_ngl]
    p = '%.4f' % (p)

    size_diff = np.abs(len(num_grids_list) - len(yfit))
    linfit = ax.plot(num_grids_list[size_diff:], yfit, alpha=0.4,
                     label='d = ' + str(d) + ', $p$ = ' + p
                           + ', $r^2$ = ' + r2,
                     linewidth=3)
    ax.plot(num_grids_list, E_error, marker='o', linestyle='None', color=linfit[0].get_color())

plt.xscale('log')
plt.yscale('log')

ax.set_xlabel("$N$", fontsize=18)
ax.set_ylabel("|Error| (au)", fontsize=18)

plt.legend(fontsize=16)
plt.title('Error in ground state vs. number of grids', fontsize=20)
plt.grid(True)
plt.gca().xaxis.grid(True, which='minor')
plt.gca().yaxis.grid(True, which='minor')

plt.show()
sys.exit(0)  # ----------------------
