import numpy as np
import functools
import matplotlib.pyplot as plt
from matplotlib import ticker
import single_electron, ext_potentials
import analytical_results
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import sys
import time
import copy

from scipy import special
from scipy import integrate
from scipy import optimize
import math
from scipy import linalg

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

# Greens function test -----------------------------------------------

N = 105
L = 18
grids = np.linspace(-L / 2, L / 2, N)
solver = single_electron.EigenSolver(grids, potential_fn=functools.partial(
    analytical_results.exp_hydro_cont_well, A=A, k=k, d=6), boundary_condition='open', n_point_stencil=3)
H = solver._h

root_list = analytical_results.exp_hydro_cont_well_roots(A,k,d=6)
energy_list = [-(1 / 8) * k ** 2 * x ** 2 for x in root_list]
th_vals = energy_list  # theoretical/analytical values
print('Htot analytical energy',th_vals)


block_matrix_size = 3
partition_size = int(N / block_matrix_size)  # must be integer

block_matrix = []
for i in range(block_matrix_size):
    row_i = []
    for j in range(block_matrix_size):
        block = H[partition_size * i:partition_size * (i + 1),
                partition_size * j:partition_size * (j + 1)]
        row_i.append(block)
    block_matrix.append(row_i)

H_block = copy.deepcopy(block_matrix)
V_block = copy.deepcopy(block_matrix)


V_block[0][0] = np.zeros((partition_size, partition_size))
V_block[0][2] = np.zeros((partition_size, partition_size))
V_block[1][1] = np.zeros((partition_size, partition_size))
V_block[2][0] = np.zeros((partition_size, partition_size))
V_block[2][2] = np.zeros((partition_size, partition_size))

H_block[0][1] = np.zeros((partition_size, partition_size))
H_block[1][0] = np.zeros((partition_size, partition_size))
H_block[1][2] = np.zeros((partition_size, partition_size))
H_block[2][1] = np.zeros((partition_size, partition_size))


H0 = np.block(H_block)
V = np.block(V_block)


#print(np.array_equal(np.block(block_matrix),H0 + V)) #should always return true!
G_0 = lambda z: np.linalg.inv(z * np.identity(N) - H0)

z_lst = np.linspace(-.1, -.8, 1000)
G_z_lst = []
for z in z_lst:
    id = np.identity(N)

    mult = np.matmul(G_0(z),V)

    G_z = np.matmul(np.linalg.inv(id - mult), G_0(z))

    #G_z_sandwich = np.matmul(np.ones(N),np.matmul(G_z,np.ones((N,1))))*(1/(N**-2))

    G_z_lst.append(1/G_z[40][40])

plt.plot(z_lst, G_z_lst, label = '$1/< G(E) > $')
label='analytic energy'
for E_th in th_vals:
    plt.axvline(x=E_th, color='red', linestyle='--', alpha=.4, label = label)
    label = ''

ax.set_xlabel("E (Ha)", fontsize=18)
ax.set_ylabel("$1/< G(E) > $", fontsize=18)

plt.legend(fontsize=16)

plt.show()
sys.exit()

