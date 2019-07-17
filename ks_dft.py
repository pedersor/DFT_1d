import numpy as np
import single_electron, dft_potentials
import math
import functools


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

    def __init__(self, grids, v_ext, v_h, xc, num_electrons=1, k_point=None, end_points=False):
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
          end_points: if true, forward/backward finite difference methods will be used
              near the boundaries to ensure the wavefunction is zero at boundaries.
              This should only be used when the grid interval is purposefully small.
              If false, all ghost points outside of the grid are set to zero. This should
              be used whenever the grid interval is sufficiently large. Setting to false
              also results in a faster computational time due to matrix symmetry.

        Raises:
          ValueError: If num_electrons is less than 1; or num_electrons is not
              an integer.
        """
        # 1d grids.
        self.k = k_point
        self.end_points = end_points
        self.grids = grids
        self.dx = get_dx(grids)
        self.num_grids = len(grids)
        # Potential on grid.
        self.v_ext = v_ext
        self.v_h = v_h
        self.xc = xc
        self.v_tot = v_ext

        if not isinstance(num_electrons, int):
            raise ValueError('num_electrons is not an integer.')
        elif num_electrons < 1:
            raise ValueError('num_electrons must be greater or equal to 1, but got %d'
                             % num_electrons)
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
    """Represents the Hamiltonian as a matrix and diagonalizes it directly.
    """

    def __init__(self, grids, v_ext, v_h, xc, num_electrons=1, k_point=None, end_points=False):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
            (num_grids,)
          num_electrons: Integer, the number of electrons in the system.
        """
        super(KS_Solver, self).__init__(grids, v_ext, v_h, xc, num_electrons, k_point, end_points)
        self.initialize_density()

    def initialize_density(self):
        # uniform density
        n_0 = self.num_electrons / (self.num_grids * self.dx) * np.ones(self.num_grids)
        self.density = n_0
        return self

    def update_v_tot(self):
        # total potential to be solved self consistently in the Kohn Sham system
        self.v_tot = functools.partial(dft_potentials.tot_KS_potential, n=self.density,
                                       v_ext=self.v_ext,
                                       v_h=self.v_h, v_xc=self.xc.v_x_exp)

        return self

    def _update_ground_state(self, solver):
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
        self.total_energy = 0.
        self.density = np.zeros(self.num_grids)
        self.kinetic_energy = 0.
        self.potential_energy = 0.
        self.eigenvalues = solver.eigenvalues
        self.wave_function = solver.wave_function

        # needs to be modified for polarized electrons
        num_unpol_states = math.floor(self.num_electrons / 2)
        for i in range(num_unpol_states):
            self.total_energy += 2 * solver.eigenvalues[i]
            self.density += 2 * (self.wave_function[i] ** 2)
            self.kinetic_energy += 2 * quadratic(solver._t_mat,
                                                 self.wave_function[i]) * self.dx

        # if odd, radical
        if self.num_electrons % 2 == 1:
            self.total_energy += solver.eigenvalues[num_unpol_states]
            self.density += (self.wave_function[num_unpol_states] ** 2)
            self.kinetic_energy += 2 * quadratic(solver._t_mat,
                                                 self.wave_function[num_unpol_states]) * self.dx

        return self

    def solve_ground_state(self):
        """Solve ground state by diagonalize the Hamiltonian matrix directly.

        Compute attributes:
        total_energy, kinetic_energy, potential_energy, density, wave_function.

        Returns:
          self
        """
        solver = single_electron.EigenSolver(self.grids, potential_fn=self.v_tot,
                                             num_electrons=self.num_electrons,
                                             end_points=True)
        solver.solve_ground_state()

        return self._update_ground_state(solver)

    def solve_self_consistent_density(self):

        density_change_integral = 1.0
        print('|Convergence error| :')
        while density_change_integral > 1e-6:
            old_density = self.density
            # solve KS system -> obtain new new density
            self.solve_ground_state()
            # update v_tot using new density
            self.update_v_tot()

            density_change_integral = np.abs(old_density - self.density).sum() * self.dx
            print(density_change_integral)

        self._solved = True
        print('converged successfully')
        print()

        # non-interacting kinetic energy
        self.T_s = self.kinetic_energy

        # external potential functional
        self.V = (self.v_ext(self.grids) * self.density).sum() * self.dx

        # Hartree integral
        self.U = .5 * (self.v_h(grids=self.grids, n=self.density) * self.density).sum() * self.dx

        # exchange energy
        self.E_x = self.xc.E_x(self.density)

        # total energy functional
        self.E_tot = self.T_s + self.V + self.U + self.E_x

        return self
