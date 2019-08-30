import ext_potentials
import numpy as np


def tot_KS_potential(grids, n, v_ext, v_h, v_xc, nUP, nDOWN):
    return v_ext(grids) + v_h(grids=grids, n=n) + v_xc(n, nUP, nDOWN)


def hartree_potential_exp(grids, n, A, k, a):
    N = len(grids)
    dx = np.abs(grids[1] - grids[0])
    v_hartree = np.zeros(N)
    for i in range(N):
        for j in range(N):
            v_hartree[i] += n[j] * (-1) * ext_potentials.exp_hydrogenic(grids[i] - grids[j], A, k, a, Z=1)
    v_hartree *= dx
    return v_hartree


class exchange_correlation_functional(object):

    def __init__(self, grids, A, k):
        self.grids = grids
        self.A = A
        self.k = k
        self.dx = (grids[-1] - grids[0]) / (len(grids) - 1)

    def set_pade_approx_params(self, n):
        '''
        parameters derived in:

        Thomas E Baker, E Miles Stoudenmire, Lucas O Wagner, Kieron Burke,
        and  Steven  R  White. One-dimensional mimicking of electronic structure:
        The case for exponentials. Physical Review B,91(23):235141, 2015.
        '''

        # these expressions are used to compute v_xc in the exponential coulomb potential case
        firstU = self.first(n, 2, -1.00077, 6.26099, -11.9041, 9.62614, -1.48334, 1)
        firstP = self.first(n, 180.891, -541.124, 651.615, -356.504, 88.0733, -4.32708, 8)
        secondU = self.second(n, 2, -1.00077, 6.26099, -11.9041, 9.62614, -1.48334, 1)
        secondP = self.second(n, 180.891, -541.124, 651.615, -356.504, 88.0733, -4.32708, 8)

        return firstU, firstP, secondU, secondP

    def v_xc_exp_up(self, n, n_up, n_down):
        # Exchange-Correlation Potential for up electrons
        # v_xc_up = d/dn_up (eps_x + eps_c)

        pi = np.pi
        firstU, firstP, secondU, secondP = self.set_pade_approx_params(n)

        v_x = -(self.A / (pi)) * (np.arctan(2 * pi * n_up / self.k))
        v_c = (self.A * (2 * firstP * (firstU ** 2) * (n_down - n_up) + (firstU ** 2) * (
                (n_down - n_up) ** 2) * secondP - 4 * (firstP ** 2) * n_down * (firstU - n_up * secondU))) / (
                      (firstP ** 2) * (firstU ** 2) * self.k)

        return v_x + v_c

    def v_xc_exp_down(self, n, n_up, n_down):
        # Exchange-Correlation Potential for down electrons
        # v_xc_down = d/dn_down (eps_x + eps_c)

        pi = np.pi
        firstU, firstP, secondU, secondP = self.set_pade_approx_params(n)

        v_x = -(self.A / (pi)) * (np.arctan(2 * pi * n_down / self.k))
        v_c = (self.A * (2 * firstP * (firstU ** 2) * (-n_down + n_up) + (firstU ** 2) * (
                (n_down - n_up) ** 2) * secondP - 4 * (firstP ** 2) * n_up * (firstU - n_down * secondU))) / (
                      (firstP ** 2) * (firstU ** 2) * self.k)

        return v_x + v_c

    def first(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
        y = np.pi * n / self.k
        return alpha + beta * (y ** (1. / 2.)) + gamma * y + delta * (y ** (3. / 2.)) + eta * (y ** 2) + sigma * (
                y ** (5. / 2.)) + nu * (np.pi * (self.k ** 2) / self.A) * (y ** 3)

    def second(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
        return (6 * (n ** 3) * (np.pi ** 4) * self.k * nu + self.A * np.sqrt(np.pi) * (
                4 * (n ** 2) * (np.pi ** (3. / 2.)) * eta + 2 * n * np.sqrt(
            np.pi) * gamma * self.k + 3 * n * np.pi * delta * np.sqrt(n / self.k) * self.k + beta * np.sqrt(
            n / self.k) * (self.k ** 2) + 5 * n * (np.pi ** 2) * ((n / self.k) ** (3. / 2.)) * self.k * sigma)) / (
                       2 * self.A * n * (self.k ** 2))

    def eps_x(self, n, zeta):
        # Exchange Energy per Length

        y = np.pi * n / self.k
        return self.A * self.k * (
                np.log(1 + (y ** 2) * ((1 + zeta) ** 2)) - 2 * y * (1 + zeta) * np.arctan(y * (1 + zeta)) + np.log(
            1 + (y ** 2) * ((-1 + zeta) ** 2)) - 2 * y * (-1 + zeta) * np.arctan(y * (-1 + zeta))) / (
                       4 * (np.pi ** 2))


    def eps_c(self, n, zeta):
        # Correlation Energy per Length

        unpol = self.corrExpression(n, 2, -1.00077, 6.26099, -11.9041, 9.62614, -1.48334, 1)
        pol = self.corrExpression(n, 180.891, -541.124, 651.615, -356.504, 88.0733, -4.32708, 8)
        return unpol + (zeta ** 2) * (pol - unpol)

    def corrExpression(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
        '''
        parameters derived in:

        Thomas E Baker, E Miles Stoudenmire, Lucas O Wagner, Kieron Burke,
        and  Steven  R  White. One-dimensional mimicking of electronic structure:
        The case for exponentials. Physical Review B,91(23):235141, 2015.
        '''

        y = np.pi * n / self.k
        return (-self.A * self.k * (y ** 2) / (np.pi ** 2)) / (
                alpha + beta * (y ** (1. / 2.)) + gamma * y + delta * (y ** (3. / 2.)) + eta * (y ** 2) + sigma * (
                y ** (5. / 2.)) + nu * (np.pi * (self.k ** 2) / self.A) * (y ** 3))


    def E_x(self, n, zeta):
        # Total Exchange Energy

        return self.eps_x(n, zeta).sum() * self.dx


    def E_c(self, n, zeta):
        # Total Correlation Energy

        return self.eps_c(n, zeta).sum() * self.dx
