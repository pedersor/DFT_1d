import HF_scf, functionals, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools

A = 1.071295
k = 1. / 2.385345

grids = np.linspace(-10, 10, 501)

v_ext = functools.partial(ext_potentials.exp_hydrogenic, A=A, k=k, a=0, Z=1)
v_h = functools.partial(functionals.hartree_potential_exp, A=A, k=k, a=0)
fock_op = functionals.fock_operator(grids=grids,A=A, k=k)

solver = HF_scf.HF_Solver(grids, v_ext=v_ext, v_h=v_h, fock_operator=fock_op, num_electrons=1)
solver.solve_self_consistent_density()

print('E_x = ', solver.E_x)
print('E = ', solver.E_tot)
