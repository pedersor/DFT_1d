# Test of LDA implementation on H_n chains

import ks_dft, functionals, ext_potentials
import matplotlib.pyplot as plt
import numpy as np
import functools

A = 1.071295
k = 1. / 2.385345

def lda_run(grids, N_e, d, Z, sym):
    # N_e: Number of Electrons
    # d: Nuclear Distance
    # Z: Nuclear Charge

    v_ext = functools.partial(ext_potentials.exp_H20, A=A, k=k, a=0, d=d, Z=Z)
    v_h = functools.partial(functionals.hartree_potential_exp, A=A, k=k, a=0)
    ex_corr = functionals.exchange_correlation_functional(grids=grids, A=A, k=k)

    solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, H_n=True, num_electrons=N_e)
    solver.solve_self_consistent_density(v_ext(grids), sym)

    return solver


if __name__ == '__main__':
    grids = np.linspace(-50, 50, 1001)

    solver = lda_run(grids=grids, N_e=20, d=4, Z=1, sym=1)
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(211)
    ax1.set_xlabel('x')
    ax1.set_ylabel('n(x)')
    plt.plot(grids, solver.density, 'r')
    plt.grid()
    ax2 = fig1.add_subplot(212)
    ax2.set_xlabel('x')
    ax2.set_ylabel('v(x)')
    plt.plot(grids, ext_potentials.exp_H20(grids=grids, A=A, k=k, a=0, d=4, Z=1), 'b')
    plt.grid()

    data = np.column_stack((grids, solver.density))
    np.savetxt('H20.txt', data)

    plt.show()