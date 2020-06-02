import blue_potentials
import numpy as np
import scipy.special as scipy
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    L = 20
    # for 1000, L = 20 using initial pt 0.018435 to minimize error of H and He
    grid_size = 1000
    grids = np.linspace(0.018435, L, grid_size)
    h = grids[1] - grids[0]

    n_r0_Z_2_erf = np.load('n_r0_Z_blue_gam_1.npy')[0]
    n_r0 = np.load('n_r0.npy')


    idx = 5
    plt.plot(grids, grids*grids*n_r0_Z_2_erf[idx])
    plt.plot(grids, grids*grids*n_r0[idx])
    plt.show()


    grid_idx = 5
    r0 = grids[grid_idx]
    print('r0', r0)

    og_sph_avg = blue_potentials.blue_helium(grids, r0, Z=0)
    erf_sph_avg = blue_potentials.blue_helium_spherical_erf(grids, r0, gam=1, Z=0)
    og_sph_avg_half = blue_potentials.blue_helium(grids, r0, Z=0, lam=0.5)

    print(erf_sph_avg)

    plt.plot(grids, og_sph_avg, label='og_sph_avg')
    plt.plot(grids, erf_sph_avg, label='erf_sph_avg')
    plt.plot(grids, og_sph_avg_half, label='og_sph_avg_half')

    # plt.legend()
    plt.show()
