import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import blue_tools


def get_n(r):
    # 3D hooke's atom radial density

    n = (2 / ((np.pi ** (3 / 2)) * (8 + 5 * (np.pi ** 0.5)))) * np.exp(
        -0.5 * (r ** 2)) * (((np.pi / 2) ** (0.5)) * (
            (7 / 4) + (0.25 * (r ** 2)) + (r + (1 / r)) * special.erf(
        r / np.sqrt(2))) + np.exp(-0.5 * (r ** 2)))

    n[0] = 0.0893193  # take the limit for r -> 0
    return n


if __name__ == '__main__':
    grids = np.linspace(0.0056, 6, 1000)

    n = get_n(grids)

    print('n integral check: ',
          np.trapz(4 * np.pi * (grids ** 2) * n, grids))

    v_H_n = blue_tools.get_v_H_n(grids, n)
    U_H = blue_tools.get_U_xc(grids, n, v_H_n)

    n_r0_sph_exact = np.load('n_r0_hookes_atom_sph_exact.npy')
    n_r0_sph_exact = np.load('n_r0_blue_gam_0.npy')

    print('n_r0_sph_exact int. check: ',
          np.trapz(4 * np.pi * (grids ** 2) * n_r0_sph_exact[33], grids))

    v_h_n_ee = blue_tools.get_v_H_n_xc(grids, n_r0_sph_exact)
    plt.plot(grids, v_h_n_ee)
    plt.show()

    V_ee = blue_tools.get_U_xc(grids, n, v_h_n_ee)

    print('Vee = ', V_ee)
    print('Uc = ', V_ee - (U_H / 2))

    # non spherical Uc
    print('non-sphericalized')
    print('Uc = ', 0.447443 - (U_H / 2))
