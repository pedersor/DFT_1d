import single_electron, functionals, ext_potentials
import numpy as np
import functools
import math


def get_dx(grids):
    """Gets the grid spacing from grids array.

    Args:
      grids: Numpy array with shape (num_grids,).

    Returns:
      Grid spacing.
    """
    return (grids[-1] - grids[0]) / (len(grids) - 1)


def quadratic(mat, x):
    """Compute the quadratic form of matrix and vector.

    Args:
      mat: matrix.
        (n, n)
      x: vector.
        (n,)

    Returns:
      output: scalar value as result of x A x.T.
    """
    return np.dot(x, np.dot(mat, x))


class SolverBase(object):
    """Base Solver for non-interacting 1d system.

    Subclasses should define solve_ground_state method.
    """

    def __init__(self, grids, v_ext, v_h, xc, num_electrons=1, k_point=None, boundary_condition="open"):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
              (num_grids,)
          v_ext: Kohn Sham external potential function taking grids as argument.
          v_h: Kohn Sham hartree potential function taking grids as argument.
          xc: exchange correlation functional class taking density as argument.
          num_electrons: integer, the number of electrons in the system. Must be
              greater or equal to 1.
          k_point: the k-point in reciprocal space used to evaluate Schrodinger Equation
              for the case of a periodic potential. It should be chosen to be within
              the first Brillouin zone.

        Raises:
          ValueError: If num_electrons is less than 1; or num_electrons is not
              an integer.
        """
        self.k = k_point
        self.boundary_condition = boundary_condition
        self.grids = grids
        self.dx = get_dx(grids)
        self.num_grids = len(grids)

        self.v_ext = v_ext
        self.v_h = v_h
        self.xc = xc
        self.v_tot_up = v_ext
        self.v_tot_down = v_ext

        self.fock_mat_up = None
        self.fock_mat_down = None

        if not isinstance(num_electrons, int):
            raise ValueError('num_electrons is not an integer.')
        elif num_electrons < 1:
            raise ValueError('num_electrons must be greater or equal to 1, but got %d' % num_electrons)
        else:
            self.num_electrons = num_electrons

        # Solver is unsolved by default.
        self._solved = False

    def is_solved(self):
        """Returns whether this solver has been solved."""
        return self._solved

    def solve_ground_state(self):
        """Solve ground state. Need to be implemented in subclasses.

        Compute attributes:
        total_energy, kinetic_energy, potential_energy, density, wave_function.

        Returns:
          self
        """
        raise NotImplementedError('Must be implemented by subclass.')


class HF_Solver(SolverBase):
    """Represents the Hamiltonian as a matrix and diagonalizes it directly."""

    def __init__(self, grids, v_ext, v_h, xc, num_electrons=1, k_point=None, boundary_condition='open'):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
            (num_grids,)
          num_electrons: Integer, the number of electrons in the system.
        """
        super(HF_Solver, self).__init__(grids, v_ext, v_h, xc, num_electrons, k_point, boundary_condition)
        self.initialize_density()

    def initialize_density(self):
        # Get number of Up/Down Electrons. All unpaired electrons are defaulted to spin-up.

        num_UP_electrons = int(self.num_electrons / 2)
        num_DOWN_electrons = int(self.num_electrons / 2)
        if self.num_electrons % 2 == 1:
            num_UP_electrons += 1

        self.num_UP_electrons = num_UP_electrons
        self.num_DOWN_electrons = num_DOWN_electrons

        # uniform density
        self.nUP = self.num_UP_electrons / (self.num_grids * self.dx) * np.ones(self.num_grids)
        self.nDOWN = self.num_DOWN_electrons / (self.num_grids * self.dx) * np.ones(self.num_grids)
        self.density = self.nUP + self.nDOWN
        self.zeta = (self.nUP - self.nDOWN) / (self.density)
        return self

    def update_v_tot_up(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_tot_up = functools.partial(functionals.tot_HF_potential, n=self.density, v_ext=self.v_ext,
                                          v_h=self.v_h)
        return self

    def update_v_tot_down(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_tot_down = functools.partial(functionals.tot_HF_potential, n=self.density, v_ext=self.v_ext,
                                            v_h=self.v_h)
        return self

    def update_fock_matrix_up(self):

        A = 1.071295
        k = 1. / 2.385345

        mat = np.zeros((self.num_grids, self.num_grids))

        for j in range(self.num_UP_electrons):

            mat_j = np.zeros((self.num_grids, self.num_grids))
            for row in range(self.num_grids):
                for column in range(self.num_grids):
                    mat_j[row, column] = ext_potentials.exp_hydrogenic(self.grids[row] - self.grids[column], A, k) * \
                                         self.wave_functionUP[j][column] * self.wave_functionUP[j][row] * self.dx

            mat += mat_j

        self.fock_mat_up = mat

        return self

    def update_fock_matrix_down(self):

        A = 1.071295
        k = 1. / 2.385345

        mat = np.zeros((self.num_grids, self.num_grids))

        for j in range(self.num_DOWN_electrons):

            mat_j = np.zeros((self.num_grids, self.num_grids))
            for row in range(self.num_grids):
                for column in range(self.num_grids):
                    mat_j[row, column] = ext_potentials.exp_hydrogenic(self.grids[row] - self.grids[column], A, k) * \
                                         self.wave_functionDOWN[j][column] * self.wave_functionDOWN[j][row] * self.dx

            mat += mat_j

        self.fock_mat_down = mat

        return self

    def get_E_x_HF(self):

        A = 1.071295
        k = 1. / 2.385345

        tot_up = 0
        for i in range(self.num_UP_electrons):
            tot_2 = 0
            for j in range(self.num_UP_electrons):
                int_tot = 0
                for x_i, x in enumerate(self.grids):

                    int_ft_of_x = np.zeros(self.num_grids)
                    for x_prime_i, x_prime in enumerate(self.grids):
                        int_ft_of_x[x_i] += ext_potentials.exp_hydrogenic(x - x_prime, A, k) * self.wave_functionUP[i][
                            x_i] * self.wave_functionUP[j][x_i] * self.wave_functionUP[i][x_prime_i] * \
                                            self.wave_functionUP[j][x_prime_i] * self.dx

                    int_tot += int_ft_of_x[x_i] * self.dx
                tot_2 += int_tot
            tot_up += tot_2

        tot_up = -.5 * tot_up

        tot_down = 0
        for i in range(self.num_DOWN_electrons):
            tot_2 = 0
            for j in range(self.num_DOWN_electrons):
                int_tot = 0
                for x_i, x in enumerate(self.grids):

                    int_ft_of_x = np.zeros(self.num_grids)
                    for x_prime_i, x_prime in enumerate(self.grids):
                        int_ft_of_x[x_i] += ext_potentials.exp_hydrogenic(x - x_prime, A, k) * \
                                            self.wave_functionDOWN[i][
                                                x_i] * self.wave_functionDOWN[j][x_i] * self.wave_functionDOWN[i][
                                                x_prime_i] * \
                                            self.wave_functionDOWN[j][x_prime_i] * self.dx

                    int_tot += int_ft_of_x[x_i] * self.dx
                tot_2 += int_tot
            tot_down += tot_2

        tot_down = -.5 * tot_down

        return tot_up + tot_down

    def _update_ground_state(self, solverUP, solverDOWN=None):
        """Helper function to solve_ground_state() method.

        Updates the attributes total_energy, wave_function, density, kinetic_energy,
        potential_enenrgy and _solved from the eigensolver's output (w, v).

        Args:
          eigenvalues: Numpy array with shape [num_eigenstates,], the eigenvalues in
              ascending order.
          eigenvectors: Numpy array with shape [num_grids, num_eigenstates], each
              column eigenvectors[:, i] is the normalized eigenvector corresponding
              to the eigenvalue eigenvalues[i].
          quadratic_function: Callable, compute the quadratic form of matrix and
              vector.

        Returns:
          self
        """

        self.kinetic_energy = 0.

        self.density = np.zeros(self.num_grids)
        self.nUP = np.zeros(self.num_grids)
        self.nDOWN = np.zeros(self.num_grids)

        self.wave_functionUP = solverUP.wave_function
        if solverDOWN is not None:
            self.wave_functionDOWN = solverDOWN.wave_function

        for i in range(self.num_UP_electrons):
            self.nUP += self.wave_functionUP[i] ** 2
            self.kinetic_energy += quadratic(solverUP._t_mat, solverUP.wave_function[i]) * self.dx

        for i in range(self.num_DOWN_electrons):
            self.nDOWN += self.wave_functionDOWN[i] ** 2
            self.kinetic_energy += quadratic(solverDOWN._t_mat, solverDOWN.wave_function[i]) * self.dx

        self.density = self.nUP + self.nDOWN
        self.zeta = (self.nUP - self.nDOWN) / (self.density)

        return self

    def solve_ground_state(self):
        """Solve ground state by diagonalizing the Hamiltonian matrix directly and separately for up and down spins.
        """

        solverUP = single_electron.EigenSolver(self.grids, potential_fn=self.v_tot_up,
                                               num_electrons=self.num_UP_electrons,
                                               boundary_condition=self.boundary_condition, fock_mat=self.fock_mat_up)
        solverUP.solve_ground_state()

        if self.num_DOWN_electrons == 0:
            return self._update_ground_state(solverUP)
        else:
            solverDOWN = single_electron.EigenSolver(self.grids, potential_fn=self.v_tot_down,
                                                     num_electrons=self.num_DOWN_electrons,
                                                     boundary_condition=self.boundary_condition,
                                                     fock_mat=self.fock_mat_down)
            solverDOWN.solve_ground_state()
            return self._update_ground_state(solverUP, solverDOWN)

    def solve_self_consistent_density(self):

        delta_E = 1.0
        first_iter = True
        while delta_E > 1e-6:
            if not first_iter:
                old_E = self.E_tot

            # solve KS system -> obtain new new density
            self.solve_ground_state()

            # update total potentials using new density
            self.update_v_tot_up()
            self.update_v_tot_down()
            self.update_fock_matrix_up()
            self.update_fock_matrix_down()

            # Non-Interacting Kinetic Energy
            self.T_s = self.kinetic_energy

            # External Potential Energy
            self.V = (self.v_ext(self.grids) * self.density).sum() * self.dx

            # Hartree Integral
            self.U = .5 * (self.v_h(grids=self.grids, n=self.density) * self.density).sum() * self.dx

            # Exchange Energy
            self.E_x = self.get_E_x_HF()

            # Total Energy
            self.E_tot = self.T_s + self.V + self.U - self.E_x

            if not first_iter:
                delta_E = np.abs(old_E - self.E_tot).sum() * self.dx
            else:
                first_iter = False

        self._solved = True

        return self
