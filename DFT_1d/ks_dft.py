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
from utils import quadratic
from scf_base import SCF_SolverBase


class KS_Solver(SCF_SolverBase):
    """KS-DFT solver for non-periodic systems."""

    def __init__(self, grids, v_ext, xc, num_electrons=1,
                 boundary_condition='open'):
        """Initialize the solver with potential function and grid.

        Args:
          xc: exchange correlation functional class, e.g. LDA.
        """
        super(KS_Solver, self).__init__(grids, v_ext, num_electrons,
                                        boundary_condition)
        self.xc = xc
        self.init_v_s()

    def init_v_s(self, v_s_up=None, v_s_down=None):
        """Initialize starting v_s_up and v_s_down. The default
        corresponds to v_hxc_up = v_hxc_down = 0. """

        if v_s_up is None and v_s_up is None:
            # default initalization, v_s = v_ext.
            self.v_s_up = self.v_ext
            self.v_s_down = self.v_ext
        else:
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

        self.phi_up = solver_up.wave_function
        self.n_up = solver_up.density
        self.kinetic_energy += solver_up.kinetic_energy
        self.eps += solver_up.total_energy

        if solver_down:
            self.phi_down = solver_down.wave_function
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
