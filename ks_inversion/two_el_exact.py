import single_electron, ext_potentials, functionals, ks_dft
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys

# plotting parameters
params = {'mathtext.default': 'default'}
plt.rcParams.update(params)
plt.rcParams['axes.axisbelow'] = True
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
fig, ax = plt.subplots()


def get_truncated_system(n, tol):
    start = 0
    for i, nx in enumerate(n):
        if nx > tol:
            start = i
            break
    end = -1
    for i, nx in enumerate(n[::-1]):
        if nx > tol:
            end = -i - 1
            break

    return start, end


def v_s_full(v_s_trunc, start, end, grids):
    N = len(grids)
    v_s_full = np.zeros(N)

    v_s_trunc = v_s_trunc - v_s_trunc[2]

    v_s_full[start + 2:end - 2] = v_s_trunc[2:-2]

    return v_s_full


def get_T_mat(grids):
    solver = single_electron.EigenSolver(grids,
                                         potential_fn=functools.partial(
                                             ext_potentials.exp_hydrogenic),
                                         boundary_condition='open',
                                         num_electrons=1)

    T_mat = solver.get_kinetic_matrix()

    return T_mat


def get_derivative_mat(grids, h):
    N = len(grids)
    mat = np.eye(N)
    idx = np.arange(N)

    A = [0, 2 / 3, -1 / 12]

    for j, A_n in enumerate(A):
        mat[idx[j:], idx[j:] - j] = -A_n
        mat[idx[:-j], idx[:-j] + j] = A_n

    return mat / h


def get_v_s_trunc(n, grids, tol):
    start, end = get_truncated_system(n, tol)
    trunc_n = n[start:end]
    trunc_grids = grids[start:end]

    T = get_T_mat(trunc_grids)
    n_sqrt = np.sqrt(trunc_n)

    v_s_trunc = - np.dot(T, np.transpose(n_sqrt)) / n_sqrt

    return trunc_grids[2:-2], v_s_trunc[2:-2]


def v_s_extension(grids, n, h, tol=10 ** (-4)):
    trunc_grids, v_s_trunc = get_v_s_trunc(n, grids, tol=tol)
    v_s_full_out = v_s_derivs(v_s_trunc, n, grids, h, tol)

    return v_s_full_out


def v_s_derivs(v_s, n, grids, h, tol, k=(1. / 2.385345)):
    start, end = get_truncated_system(n, tol=tol)
    N = len(grids)
    v_s_full = np.zeros(N)

    # fdm_1_coeffs = [-25 / 12, 4, -3, 4 / 3, -1 / 4]
    fdm_1_coeffs = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
    fdm_1_coeffs = np.asarray(fdm_1_coeffs)

    center = 0

    left_1_deriv = np.sum(v_s[0 + center:5 + center] * fdm_1_coeffs) / h
    right_1_deriv = np.sum(
        v_s[-1 - 5 - center:-1 - center] * -fdm_1_coeffs[::-1]) / h

    A_left = left_1_deriv / (k * np.exp(k * (grids[start + 2] + 2 * h)))

    v_s = v_s - v_s[2] + left_1_deriv / k
    A_right = v_s[-3] / np.exp(-k * (grids[end - 3] - 2 * h))

    v_s_full[start + 2 + 2:end - 2 - 2] = v_s[2:-2]

    v_s_full[0:start + 2 + 2] = A_left * np.exp(k * grids[0: start + 2 + 2])
    v_s_full[end - 2 - 2:] = A_right * np.exp(-k * grids[end - 2 - 2:])

    return v_s_full


if __name__ == '__main__':
    # test using H2 data

    dir = 'H2_data'
    densities = np.load(dir + "/densities.npy")
    locations = np.load(dir + "/locations.npy")
    potentials_ext = np.load(dir + "/potentials.npy")
    total_energies = np.load(dir + "/total_energies.npy")

    h = 0.08
    grids = np.arange(-256, 257) * h

    v_ext = potentials_ext[70]

    v_s_out = []
    for n in densities:
        v_s_out.append(v_s_extension(grids, n, h))

    v_s_out = np.asarray(v_s_out)
    np.save(dir + "/exact_v_s_extrapolated.npy", v_s_out)

    # v_s_out = np.load(dir + "/exact_v_s_extrapolated.npy")

    plt.plot(grids, v_s_out[20], label="R = " + str(21 * h))
    plt.plot(grids, v_s_out[40], label="R = " + str(41 * h)[:4])
    plt.plot(grids, v_s_out[60], label="R = " + str(61 * h))
    plt.plot(grids, v_s_out[70], label="R = " + str(71 * h))

    plt.plot(grids, v_ext, label="$V_{ext}$: R = " + str(71 * h))

    plt.xlabel("x", fontsize=18)
    plt.ylabel("$v_s$", fontsize=18)

    plt.legend(fontsize=16)
    plt.show()
    sys.exit()
