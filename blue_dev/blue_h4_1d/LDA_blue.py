import single_electron, ext_potentials, blue_potentials
import numpy as np
import functools
import sys


h = 0.08
grids = np.arange(-256, 257) * h
potentials = np.load("H4_data/potentials.npy")


# gets cusp condition right (previously off by factor of 2)
new_cusp_cond = 0.5

n_r0_R = []
for i, potential in enumerate(potentials):
    print(i, flush=True)
    n_r0 = []
    for r0 in grids:
        solver = single_electron.EigenSolver(grids,
                                             potential_fn=functools.partial(
                                                 blue_potentials.pure_blue_arb_pot_1d,
                                                 pot=potential, r0=r0,
                                                 lam=new_cusp_cond),
                                             boundary_condition='open',
                                             num_electrons=1)
        solver.solve_ground_state()

        # E0 = solver.eigenvalues[0]
        # print("E0 = ", E0)

        n = solver.density

        n_r0.append(n)

    n_r0 = np.asarray(n_r0)
    n_r0_R.append(n_r0)

n_r0_R = np.asarray(n_r0_R)
np.save('n_r0_R_half.npy', n_r0_R)
