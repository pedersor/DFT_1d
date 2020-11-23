import single_electron, functionals
import numpy as np
import functools
import math
import matplotlib.pyplot as plt


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
        self._initialize_density()
        self.set_energy_tol_threshold()

    def set_energy_tol_threshold(self, energy_tol_threshold=1e-4):
        self.energy_tol_threshold = energy_tol_threshold
        return self

    def _initialize_density(self):
        # Get number of Up/Down Electrons. All unpaired electrons are defaulted to spin-up.

        num_up_electrons = int(self.num_electrons / 2)
        num_down_electrons = int(self.num_electrons / 2)
        if self.num_electrons % 2 == 1:
            num_up_electrons += 1

        self.num_up_electrons = num_up_electrons
        self.num_down_electrons = num_down_electrons

        return self

    def _update_v_tot_up(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_tot_up = functools.partial(functionals.tot_KS_potential,
                                          n=self.density, n_up=self.n_up,
                                          n_down=self.n_down, v_ext=self.v_ext,
                                          v_h=self.v_h,
                                          v_xc=self.xc.v_xc_exp_up)
        return self

    def _update_v_tot_down(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_tot_down = functools.partial(functionals.tot_KS_potential,
                                            n=self.density, n_up=self.n_up,
                                            n_down=self.n_down,
                                            v_ext=self.v_ext,
                                            v_h=self.v_h,
                                            v_xc=self.xc.v_xc_exp_down)
        return self

    def _update_ground_state(self, solver_up, solver_down=None):
        """Helper function to _solve_ground_state() method.

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

        self.kinetic_energy = 0
        self.eps = 0

        self.n_up = solver_up.density
        self.kinetic_energy += solver_up.kinetic_energy
        self.eps += solver_up.total_energy

        if solver_down:
            self.n_down = solver_down.density
            self.kinetic_energy += solver_down.kinetic_energy
            self.eps += solver_down.total_energy

        self.density = self.n_up + self.n_down
        self.zeta = (self.n_up - self.n_down) / (self.density)

        return self

    def _solve_ground_state(self):
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

    def solve_self_consistent_density(self, v_ext, mixing_param=0.3, verbose=0):
        """

        Args:
            mixing_param: linear mixing parameter, where 0.0 denotes no mixing.
            verbose: convergence debug printing.

        Returns:
            self.
        """

        # TODO: use prev_densities for DIIS mixing
        prev_densities = []

        final_energy = 1E100
        converged = False
        while not converged:
            # solve KS system -> obtain new density
            self._solve_ground_state()

            # update total potentials using new density
            self._update_v_tot_up()
            self._update_v_tot_down()

            if (np.abs(self.eps - final_energy) < self.energy_tol_threshold):
                converged = True
                self._solved = True

            final_energy = self.eps
            if prev_densities and mixing_param:
                self.density = (1 - mixing_param) * self.density + \
                               mixing_param * prev_densities[-1]

            prev_densities.append(self.density)

            if verbose == 1 or verbose == 2:
                print("i = " + str(len(prev_densities)) + ": eps = " + str(
                    final_energy))
            if verbose == 2:
                plt.plot(self.grids, prev_densities[-1])
                plt.show()

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

        return self
