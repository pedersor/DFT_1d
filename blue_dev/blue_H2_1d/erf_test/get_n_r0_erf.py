import os
import sys

# linux cluster path
sys.path.append('/DFS-B/DATA/burke/pedersor/Kohn_Sham_DFT_1d')
sys.path.append('/DFS-B/DATA/burke/pedersor/Kohn_Sham_DFT_1d/blue_dev')
sys.path.append(
    '/DFS-B/DATA/burke/pedersor/Kohn_Sham_DFT_1d/blue_dev/blue_H2_1d/corrected_dissociation_curve/')

import numpy as np
import functools
import single_electron, functionals, ext_potentials
import blue_potentials


def get_n_r0(grids, densities, potentials):
    n_r0 = []
    for i, pot in enumerate(potentials):
        print(i, flush=True)
        n_r0_R = []
        for j, r0 in enumerate(grids):
            blue_pot = functools.partial(blue_potentials.blue_1d_H2_erf,
                                         r0=r0, n_r=densities[i][j], pot=pot)
            solver = single_electron.EigenSolver(grids,
                                                 potential_fn=blue_pot)

            solver.solve_ground_state()
            n_r0_R.append(solver.density)
        n_r0_R = np.asarray(n_r0_R)
        n_r0.append(n_r0_R)

    n_r0 = np.asarray(n_r0)
    return n_r0


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h

    densities = np.load('../H2_data/densities.npy')
    potentials = np.load('../H2_data/potentials.npy')

    n_r0 = get_n_r0(grids, densities, potentials)

    np.save("n_r0_1D_H2_erf_gam_1or_s.npy", n_r0)
