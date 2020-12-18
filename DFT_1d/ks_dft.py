"""
Kohn-Sham DFT solver
####################

**Summary** 
    This is the summary

.. moduleauthor::
    EXAMPLE <Example@university.edu> <https://dft.uci.edu/> ORCID: `000-0000-0000-0000 <https://orcid.org/0000-0000-0000-0000>`_

.. todo::

    * Authors?
    * Comments in KS solver funciton should be in doc format
    * *solve_self_consistent_density* needs summary sentence
    * Clean out unused example rst content in here.
    * Linting?
"""

import non_interacting_solver, functionals
import numpy as np
import functools
import matplotlib.pyplot as plt
from utils import get_dx, quadratic


class SolverBase:
    """Base Solver for non-interacting Kohn-Sham (KS) 1d systems."""

    def __init__(self, grids, v_ext, xc, num_electrons=1,
                 boundary_condition="open"):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
              (num_grids,)
          v_ext: Kohn Sham external potential function taking grids as argument.
          xc: exchange correlation functional class taking density as argument.
          num_electrons: integer, the number of electrons in the system. Must be
              greater or equal to 1.

        Raises:
          ValueError: If num_electrons is less than 1; or num_electrons is not
              an integer.
        """
        self.boundary_condition = boundary_condition
        self.grids = grids
        self.dx = get_dx(grids)

        self.v_ext = v_ext
        self.xc = xc

        if not isinstance(num_electrons, int):
            raise ValueError('num_electrons is not an integer.')
        elif num_electrons < 1:
            raise ValueError(
                'num_electrons must be greater or equal to 1, but got %d' % num_electrons)
        else:
            self.num_electrons = num_electrons

        # Solver is not converged by default.
        self._converged = False
        self._init_default_spin_config()
        self.set_energy_tol_threshold()

    def set_energy_tol_threshold(self, energy_tol_threshold=1e-4):
        self.energy_tol_threshold = energy_tol_threshold
        return self

    def _init_default_spin_config(self):
        """Default spin configuration: all up/down spins are paired if
        possible. All unpaired electrons are defaulted to spin-up.
        """

        num_up_electrons = self.num_electrons // 2
        num_down_electrons = self.num_electrons // 2
        if self.num_electrons % 2 == 1:
            num_up_electrons += 1

        self.num_up_electrons = num_up_electrons
        self.num_down_electrons = num_down_electrons

        return self

    def is_converged(self):
        """Returns whether this solver has been solved."""
        return self._converged


class KS_Solver(SolverBase):
    """KS-DFT solver for non-periodic systems."""

    def __init__(self, grids, v_ext, xc, num_electrons=1,
                 boundary_condition='open'):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
            (num_grids,)
          num_electrons: Integer, the number of electrons in the system.
        """
        super(KS_Solver, self).__init__(grids, v_ext, xc, num_electrons,
                                        boundary_condition)
        self.init_v_s(v_ext, v_ext)

    def init_v_s(self, v_s_up, v_s_down):
        """Initialize starting v_s_up and v_s_down. The default
        corresponds to v_hxc_up = v_hxc_down = 0. """

        self.v_s_up = v_s_up
        self.v_s_down = v_s_down
        return self

    def _update_v_s_up(self):
        """Total up spin potential to be solved self consistently in the
        KS system.
        """

        self.v_s_up = functools.partial(self.xc.v_s_up,
                                        n=self.density, n_up=self.n_up,
                                        n_down=self.n_down, v_ext=self.v_ext,
                                        v_xc_up=self.xc.v_xc_up)
        return self

    def _update_v_s_down(self):
        """Total down spin potential to be solved self consistently in the
        KS system.
        """

        self.v_s_down = functools.partial(self.xc.v_s_down,
                                          n=self.density, n_up=self.n_up,
                                          n_down=self.n_down,
                                          v_ext=self.v_ext,
                                          v_xc_down=self.xc.v_xc_down)
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
        else:
            self.n_down = 0

        self.density = self.n_up + self.n_down
        self.zeta = (self.n_up - self.n_down) / (self.density)

        return self

    def _solve_ground_state(self):
        """Solve ground state by diagonalizing the Hamiltonian matrix directly
        and separately for up and down spins.
        """

        solver_up = non_interacting_solver.EigenSolver(self.grids,
                                                       potential_fn=self.v_s_up,
                                                       num_electrons=self.num_up_electrons,
                                                       boundary_condition=self.boundary_condition)
        solver_up.solve_ground_state()

        if self.num_down_electrons == 0:
            return self._update_ground_state(solver_up)
        else:
            solver_down = non_interacting_solver.EigenSolver(self.grids,
                                                             potential_fn=self.v_s_down,
                                                             num_electrons=self.num_down_electrons,
                                                             boundary_condition=self.boundary_condition)
            solver_down.solve_ground_state()
            return self._update_ground_state(solver_up, solver_down)

    def solve_self_consistent_density(self, mixing_param=0.3, verbose=0):
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
            self._update_v_s_up()
            self._update_v_s_down()

            if (np.abs(self.eps - final_energy) < self.energy_tol_threshold):
                converged = True
                self._converged = True

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
        v_h = self.xc.v_h()
        self.U = .5 * (v_h(grids=self.grids,
                           n=self.density) * self.density).sum() * self.dx

        # Exchange Energy
        self.E_x = self.xc.get_E_x(self.density, self.zeta)

        # Correlation Energy
        self.E_c = self.xc.get_E_c(self.density, self.zeta)

        # Total Energy
        self.E_tot = self.T_s + self.V + self.U + self.E_x + self.E_c

        return self
