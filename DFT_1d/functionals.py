"""
Functionals
###########

**Summary** 
    Defined grid based exchange-correlation functionals, fock operator, and potentials for 1D systems.

.. moduleauthor::
    EXAMPLE <Example@university.edu> <https://dft.uci.edu/> ORCID: `000-0000-0000-0000 <https://orcid.org/0000-0000-0000-0000>`_


.. todo::

    * Authors? -RJM
    * Check summary written by RJM. -RJM
    * Docs might need love; judgement call. -RJM
"""

import ext_potentials
import constants
import numpy as np
import functools


def tot_KS_potential(grids, n, v_ext, v_h, v_xc, n_up, n_down):
    return v_ext(grids) + v_h(grids=grids, n=n) + v_xc(n, n_up, n_down)


def tot_HF_potential(grids, n, v_ext, v_h):
    return v_ext(grids) + v_h(grids=grids, n=n)


def hartree_potential(grids, n, v_ee=functools.partial(
    ext_potentials.exp_hydrogenic)):
    N = len(grids)
    dx = np.abs(grids[1] - grids[0])
    v_H = np.zeros(N)
    for i in range(N):
        for j in range(N):
            v_H[i] += n[j] * (-1) * v_ee(grids[i] - grids[j])
    v_H *= dx
    return v_H


class fock_operator(object):
    def __init__(self, grids, A=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
                 k=constants.EXPONENTIAL_COULOMB_KAPPA):
        self.grids = grids
        self.num_grids = len(grids)
        self.A = A
        self.k = k
        self.dx = (grids[-1] - grids[0]) / (len(grids) - 1)

    def update_fock_matrix(self, wave_function):
        # fock matrix will be implemented as fock operator,
        # see RP logbook 9/6/19

        num_electrons = len(wave_function)
        mat = np.zeros((self.num_grids, self.num_grids))

        for j in range(num_electrons):

            mat_j = np.zeros((self.num_grids, self.num_grids))
            for row in range(self.num_grids):
                for column in range(self.num_grids):
                    mat_j[row, column] = ext_potentials.exp_hydrogenic(
                        self.grids[row] - self.grids[column], self.A,
                        self.k) * \
                                         wave_function[j][column] * \
                                         wave_function[j][row] * self.dx

            mat += mat_j

        return mat

    def get_E_x(self, wave_function):
        # obtain E_x 'exactly' from double integral over HF orbitals
        num_electrons = len(wave_function)

        E_x = 0
        for i in range(num_electrons):
            outer_int_tot = 0
            for j in range(num_electrons):
                inner_int_tot = 0
                for x_i, x in enumerate(self.grids):
                    int_fn_of_x = np.zeros(self.num_grids)
                    for x_prime_i, x_prime in enumerate(self.grids):
                        int_fn_of_x[x_i] += -1 * ext_potentials.exp_hydrogenic(
                            x - x_prime, self.A, self.k) * \
                                            wave_function[i][x_i] * \
                                            wave_function[j][x_i] * \
                                            wave_function[i][
                                                x_prime_i] * \
                                            wave_function[j][
                                                x_prime_i] * self.dx

                    inner_int_tot += int_fn_of_x[x_i] * self.dx
                outer_int_tot += inner_int_tot
            E_x += outer_int_tot

        E_x = -.5 * E_x
        return E_x


class exchange_correlation_functional(object):

    def __init__(self, grids, A=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
                 k=constants.EXPONENTIAL_COULOMB_KAPPA):
        self.grids = grids
        self.A = A
        self.k = k
        self.dx = (grids[-1] - grids[0]) / (len(grids) - 1)

    def set_pade_approx_params(self, n):
        """
        Parameters are derived in [Baker2015]_.
        """

        # these expressions are used to compute v_xc in the exponential coulomb potential case
        first_U = self.first(n, 2, -1.00077, 6.26099, -11.9041, 9.62614,
                             -1.48334, 1)
        first_P = self.first(n, 180.891, -541.124, 651.615, -356.504, 88.0733,
                             -4.32708, 8)
        second_U = self.second(n, 2, -1.00077, 6.26099, -11.9041, 9.62614,
                               -1.48334, 1)
        second_P = self.second(n, 180.891, -541.124, 651.615, -356.504, 88.0733,
                               -4.32708, 8)

        return first_U, first_P, second_U, second_P

    def first(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
        y = np.pi * n / self.k
        return alpha + beta * (y ** (1. / 2.)) + gamma * y + delta * (
                y ** (3. / 2.)) + eta * (y ** 2) + sigma * (
                       y ** (5. / 2.)) + nu * (
                       np.pi * (self.k ** 2) / self.A) * (y ** 3)

    def second(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
        return (6 * (n ** 3) * (np.pi ** 4) * self.k * nu + self.A * np.sqrt(
            np.pi) * (
                        4 * (n ** 2) * (
                        np.pi ** (3. / 2.)) * eta + 2 * n * np.sqrt(
                    np.pi) * gamma * self.k + 3 * n * np.pi * delta * np.sqrt(
                    n / self.k) * self.k + beta * np.sqrt(
                    n / self.k) * (self.k ** 2) + 5 * n * (np.pi ** 2) * (
                                (n / self.k) ** (
                                3. / 2.)) * self.k * sigma)) / (
                       2 * self.A * n * (self.k ** 2))

    def v_xc_exp_up(self, n, n_up, n_down):
        # Exchange-Correlation Potential for up electrons
        # v_xc_up = d/dn_up (eps_x + eps_c)

        pi = np.pi
        first_U, first_P, second_U, second_P = self.set_pade_approx_params(n)

        v_x = -(self.A / (pi)) * (np.arctan(2 * pi * n_up / self.k))
        v_c = (self.A * (2 * first_P * (first_U ** 2) * (n_down - n_up) + (
                first_U ** 2) * (
                                 (n_down - n_up) ** 2) * second_P - 4 * (
                                 first_P ** 2) * n_down * (
                                 first_U - n_up * second_U))) / (
                      (first_P ** 2) * (first_U ** 2) * self.k)

        return v_x + v_c

    def v_xc_exp_down(self, n, n_up, n_down):
        # Exchange-Correlation Potential for down electrons
        # v_xc_down = d/dn_down (eps_x + eps_c)

        pi = np.pi
        first_U, first_P, second_U, second_P = self.set_pade_approx_params(n)

        v_x = -(self.A / (pi)) * (np.arctan(2 * pi * n_down / self.k))
        v_c = (self.A * (2 * first_P * (first_U ** 2) * (-n_down + n_up) + (
                first_U ** 2) * (
                                 (n_down - n_up) ** 2) * second_P - 4 * (
                                 first_P ** 2) * n_up * (
                                 first_U - n_down * second_U))) / (
                      (first_P ** 2) * (first_U ** 2) * self.k)

        return v_x + v_c

    def e_x(self, n, zeta):
        # Exchange Energy per Length

        y = np.pi * n / self.k
        return self.A * self.k * (
                np.log(1 + (y ** 2) * ((1 + zeta) ** 2)) - 2 * y * (
                1 + zeta) * np.arctan(y * (1 + zeta)) + np.log(
            1 + (y ** 2) * ((-1 + zeta) ** 2)) - 2 * y * (
                        -1 + zeta) * np.arctan(y * (-1 + zeta))) / (
                       4 * (np.pi ** 2))

    def e_c(self, n, zeta):
        # Correlation Energy per Length

        unpol = self.corrExpression(n, 2, -1.00077, 6.26099, -11.9041, 9.62614,
                                    -1.48334, 1)
        pol = self.corrExpression(n, 180.891, -541.124, 651.615, -356.504,
                                  88.0733, -4.32708, 8)
        return unpol + (zeta ** 2) * (pol - unpol)

    def corrExpression(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
        """
        Parameters are derived in [Baker2015]_
        """

        y = np.pi * n / self.k
        return (-self.A * self.k * (y ** 2) / (np.pi ** 2)) / (
                alpha + beta * (y ** (1. / 2.)) + gamma * y + delta * (
                y ** (3. / 2.)) + eta * (y ** 2) + sigma * (
                        y ** (5. / 2.)) + nu * (
                        np.pi * (self.k ** 2) / self.A) * (y ** 3))

    def get_E_x(self, n, zeta):
        # Total Exchange Energy

        return self.e_x(n, zeta).sum() * self.dx

    def get_E_c(self, n, zeta):
        # Total Correlation Energy

        return self.e_c(n, zeta).sum() * self.dx
