import single_electron, functionals
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

    def __init__(self, grids, v_ext, v_h, xc, num_electrons=1, k_point=None,
                 boundary_condition="open"):
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

        if not isinstance(num_electrons, int):
            raise ValueError('num_electrons is not an integer.')
        elif num_electrons < 1:
            raise ValueError(
                'num_electrons must be greater or equal to 1, but got %d' % num_electrons)
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


class KS_Solver(SolverBase):
    """Represents the Hamiltonian as a matrix and diagonalizes it directly."""

    def __init__(self, grids, v_ext, v_h, xc, num_electrons=1,
                 k_point=None, boundary_condition='open'):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
            (num_grids,)
          num_electrons: Integer, the number of electrons in the system.
        """
        super(KS_Solver, self).__init__(grids, v_ext, v_h, xc, num_electrons,
                                        k_point, boundary_condition)
        self.initialize_density()
        self.set_energy_tol_threshold()

    def set_energy_tol_threshold(self, energy_tol_threshold = 1e-6):
        self.energy_tol_threshold = energy_tol_threshold
        return self

    def initialize_density(self):
        # Get number of Up/Down Electrons. All unpaired electrons are defaulted to spin-up.

        num_up_electrons = int(self.num_electrons / 2)
        num_down_electrons = int(self.num_electrons / 2)
        if self.num_electrons % 2 == 1:
            num_up_electrons += 1

        self.num_up_electrons = num_up_electrons
        self.num_down_electrons = num_down_electrons

        # uniform density (unused)
        self.n_up = self.num_up_electrons / (
                self.num_grids * self.dx) * np.ones(
            self.num_grids)
        self.n_down = self.num_down_electrons / (
                self.num_grids * self.dx) * np.ones(self.num_grids)
        self.density = self.n_up + self.n_down
        self.zeta = (self.n_up - self.n_down) / (self.density)

        return self

    def update_v_tot_up(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_tot_up = functools.partial(functionals.tot_KS_potential,
                                          n=self.density, n_up=self.n_up,
                                          n_down=self.n_down, v_ext=self.v_ext,
                                          v_h=self.v_h,
                                          v_xc=self.xc.v_xc_exp_up)
        return self

    def update_v_tot_down(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_tot_down = functools.partial(functionals.tot_KS_potential,
                                            n=self.density, n_up=self.n_up,
                                            n_down=self.n_down,
                                            v_ext=self.v_ext,
                                            v_h=self.v_h,
                                            v_xc=self.xc.v_xc_exp_down)
        return self

    def _update_ground_state(self, solver_up, solver_down=None):
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
        self.n_up = np.zeros(self.num_grids)
        self.n_down = np.zeros(self.num_grids)

        self.wave_function_up = solver_up.wave_function
        if solver_down is not None:
            self.wave_function_down = solver_down.wave_function

        for i in range(self.num_up_electrons):
            self.n_up += self.wave_function_up[i] ** 2
            self.kinetic_energy += quadratic(solver_up._t_mat,
                                             solver_up.wave_function[
                                                 i]) * self.dx

        for i in range(self.num_down_electrons):
            self.n_down += self.wave_function_down[i] ** 2
            self.kinetic_energy += quadratic(solver_down._t_mat,
                                             solver_down.wave_function[
                                                 i]) * self.dx

        self.density = self.n_up + self.n_down
        self.zeta = (self.n_up - self.n_down) / (self.density)

        return self

    def solve_ground_state(self):
        """Solve ground state by diagonalizing the Hamiltonian matrix directly and separately for up and down spins.
        """

        solver_up = single_electron.EigenSolver(self.grids,
                                               potential_fn=self.v_tot_up,
                                               num_electrons=self.num_up_electrons,
                                               boundary_condition=self.boundary_condition)
        solver_up.solve_ground_state()

        if self.num_down_electrons == 0:
            return self._update_ground_state(solver_up)
        else:
            solver_down = single_electron.EigenSolver(self.grids,
                                                     potential_fn=self.v_tot_down,
                                                     num_electrons=self.num_down_electrons,
                                                     boundary_condition=self.boundary_condition)
            solver_down.solve_ground_state()
            return self._update_ground_state(solver_up, solver_down)

    def solve_self_consistent_density(self, v_ext, sym):

        self.densityList = []

        self.cuspList = []
        for i in range(len(v_ext) - 1):
            if v_ext[i - 1] >= v_ext[i] and v_ext[i + 1] >= v_ext[i]:
                self.cuspList.append(i)

        delta_E = 1.0
        first_iter = True
        while delta_E >= self.energy_tol_threshold:
            if not first_iter:
                old_E = self.E_tot

            # solve KS system -> obtain new density
            self.solve_ground_state()

            # update total potentials using new density
            self.update_v_tot_up()
            self.update_v_tot_down()

            # perturb spin up/down densities to break symmetry
            if first_iter == True:
                for i in range(0, len(self.cuspList), 2):
                    for j in range(0, 10, 1):
                        self.n_up[self.cuspList[i] - j] *= sym
                        self.n_up[self.cuspList[i] + j] *= sym
                        self.n_down[self.cuspList[i] - j] *= 1. / sym
                        self.n_down[self.cuspList[i] + j] *= 1. / sym
                    self.n_up[self.cuspList[i]] *= 1. / sym
                    self.n_down[self.cuspList[i]] *= sym
                for i in range(1, len(self.cuspList), 2):
                    for j in range(0, 10, 1):
                        self.n_up[self.cuspList[i] - j] *= 1. / sym
                        self.n_up[self.cuspList[i] + j] *= 1. / sym
                        self.n_down[self.cuspList[i] - j] *= sym
                        self.n_down[self.cuspList[i] + j] *= sym
                    self.n_up[self.cuspList[i]] *= sym
                    self.n_down[self.cuspList[i]] *= 1. / sym
                self.density = self.n_up + self.n_down

            self.densityList.append(self.density)

            # Non-Interacting Kinetic Energy
            self.T_s = self.kinetic_energy

            # External Potential Energy
            self.V = (self.v_ext(self.grids) * self.density).sum() * self.dx

            # Hartree Integral
            self.U = .5 * (self.v_h(grids=self.grids,
                                    n=self.density) * self.density).sum() * self.dx

            # Exchange Energy
            self.E_x = self.xc.get_E_x(self.density, self.zeta)

            # Correlation Energy
            self.E_c = self.xc.get_E_c(self.density, self.zeta)

            # Total Energy
            self.E_tot = self.T_s + self.V + self.U + self.E_x + self.E_c

            if not first_iter:
                delta_E = np.abs(old_E - self.E_tot).sum() * self.dx
                # print("delta_E = ", delta_E)
            else:
                first_iter = False

        self._solved = True

        return self
