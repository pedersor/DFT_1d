import ks_dft, ext_potentials, dft_potentials

import numpy as np
import functools

A = 1.071295
k_inv = 2.385345
k = 1. / k_inv

grids = np.linspace(-10, 10, 200)

v_ext = functools.partial(ext_potentials.exp_hydrogenic, A=A, k=k, a=0, Z=2)
v_h = functools.partial(dft_potentials.hartree_potential_exp, A=A, k=k, a=0)
ex_corr = dft_potentials.exchange_correlation_functional(grids=grids, A=A, k=k)

solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=2, end_points=True)
solver.solve_self_consistent_density()

print("T_s =", solver.T_s)

# external potential functional
print("V =", solver.V)

# Hartree integral
print("U =", solver.U)

# exchange energy
print("E_x =", solver.E_x)

# total energy functional
print("E =", solver.E_tot)
