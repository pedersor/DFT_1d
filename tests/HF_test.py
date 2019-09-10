import HF_scf, dft_potentials, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools

A = 1.071295
k = 1. / 2.385345

grids = np.linspace(-10, 10, 501)

v_ext = functools.partial(ext_potentials.exp_hydrogenic, A=A, k=k, a=0, Z=4)
v_h = functools.partial(dft_potentials.hartree_potential_exp, A=A, k=k, a=0)
ex_corr = dft_potentials.exchange_correlation_functional(grids=grids, A=A, k=k)

solver = HF_scf.HF_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=3)
solver.solve_self_consistent_density()

print('Ex = ', solver.E_x)
print('E = ', solver.E_tot)
