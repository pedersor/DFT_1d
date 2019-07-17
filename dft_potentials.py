import numpy as np
import ext_potentials


def tot_KS_potential(grids, n, v_ext, v_h, v_xc):
    return v_ext(grids) + v_h(grids=grids, n=n) + v_xc(n)


def hartree_potential_exp(grids, n, A, k, a):
    N = len(grids)
    dx = np.abs(grids[1] - grids[0])
    v_hartree = np.zeros(N)
    for i in range(N):
        for j in range(N):
            v_hartree[i] += n[j] * (-1) * ext_potentials.exp_hydrogenic(grids[i] - grids[j], A, k,
                                                                        a, Z=1)
    v_hartree *= dx
    return v_hartree


class exchange_correlation_functional(object):
    def __init__(self, grids, A, k):
        self.grids = grids
        self.A = A
        self.k = k
        self.dx = (grids[-1] - grids[0]) / (len(grids) - 1)

    # x potential
    def v_x_exp(self, density):
        pi = np.pi
        return -2 * np.arctan(pi * density / self.k) * self.A / (2 * pi)

    # exchange energy per length
    def eps_x(self, density):
        pi = np.pi
        y = pi * density / self.k
        return self.A * self.k * (np.log(1 + y ** 2) - 2 * y * np.arctan(y)) / (2 * (pi ** 2))

    # total exchange energy
    def E_x(self, density):
        return self.eps_x(density).sum() * self.dx
