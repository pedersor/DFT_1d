import single_electron
import ext_potentials
import blue_potentials
import functionals
import ks_dft

import numpy as np
import functools
import matplotlib.pyplot as plt
import sys

h = 0.08
grids = np.arange(-256, 257) * h
potentials = np.load("h4_data/potentials.npy")
densities = np.load("h4_data/densities.npy")

# testing
n = densities[74]
potential = potentials[74]
r0 = grids[100]

A = 1.071295
k = 1. / 2.385345

v_ext = functools.partial(blue_potentials.pure_blue_arb_pot_1d,
                          pot=potential, r0=r0,
                          lam=1)

plt.plot(grids, n)
plt.plot(grids, v_ext(grids))
plt.show()

v_h = functools.partial(functionals.hartree_potential)
ex_corr = functionals.exchange_correlation_functional(grids=grids, A=A, k=k)

solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr,
                          num_electrons=3)
solver.solve_self_consistent_density(v_ext=v_ext(grids), mixing_param=0.4,
                                     verbose=1)

plt.plot(grids, solver.density)
plt.show()
