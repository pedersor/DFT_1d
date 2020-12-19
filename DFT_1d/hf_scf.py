"""
Hartree-Fock DFT solver
#######################

**Summary** 
    This is the summary

.. moduleauthor::
    EXAMPLE <Example@university.edu> <https://dft.uci.edu/> ORCID: `000-0000-0000-0000 <https://orcid.org/0000-0000-0000-0000>`_

.. todo::

    * redo in ks_dft format.
    * BaseSCF good idea? use for both ks and hf..
    * Authors?
    * Comments in HF solver funciton should be in doc format
    * *solve_self_consistent_density* needs summary sentence
    * Clean out unused example rst content in here.
    * linting?
"""

import non_interacting_solver, functionals
import numpy as np
import functools
import math
from utils import get_dx, quadratic
from scf_base import SCF_SolverBase


class HF_Solver(SCF_SolverBase):
    """Represents the Hamiltonian as a matrix and diagonalizes it directly."""

    def __init__(self, grids, v_ext, hf, num_electrons=1,
                 boundary_condition='open'):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
            (num_grids,)
          num_electrons: Integer, the number of electrons in the system.
        """
        super(HF_Solver, self).__init__(grids, v_ext, num_electrons,
                                        boundary_condition)

        self.hf = hf

        self.num_grids = len(grids)

        self.init_v_eff()

    def init_v_eff(self, v_eff_up=None, v_eff_down=None, fock_mat_up=None,
                   fock_mat_down=None):
        """Initialize starting v_eff_up and v_eff_down. """

        if v_eff_up is None and v_eff_down is None and fock_mat_up is None and fock_mat_down is None:
            # default initialization, v_eff = v_ext
            self.fock_mat_up = None
            self.fock_mat_down = None
            self.v_eff_up = self.v_ext
            self.v_eff_down = self.v_ext
        else:
            self.fock_mat_up = fock_mat_up
            self.fock_mat_down = fock_mat_down
            self.v_eff_up = v_eff_up
            self.v_eff_down = v_eff_down

        return self

    def _update_v_eff_up(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_eff_up = functools.partial(self.hf.v_hf, n=self.density,
                                          v_ext=self.v_ext)
        return self

    def _update_v_eff_down(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_eff_down = functools.partial(self.hf.v_hf, n=self.density,
                                            v_ext=self.v_ext)
        return self

    def _update_fock_matrix_up(self):
        self.fock_mat_up = self.hf.update_fock_matrix(
            wave_function=self.phi_up[:self.num_up_electrons])

        return self

    def _update_fock_matrix_down(self):
        if self.num_down_electrons == 0:
            return self
        else:
            self.fock_mat_down = self.hf.update_fock_matrix(
                wave_function=self.phi_down[:self.num_down_electrons])

            return self

    def get_E_x_HF(self):
        if self.num_down_electrons == 0:
            return self.hf.get_E_x(
                wave_function=self.phi_up[:self.num_up_electrons])
        else:
            E_x_up = self.hf.get_E_x(
                wave_function=self.phi_up[:self.num_up_electrons])
            E_x_down = self.hf.get_E_x(
                wave_function=self.phi_down[:self.num_down_electrons])
            return E_x_up + E_x_down

    def _solve_ground_state(self, first_iter, sym):
        """Solve ground state by diagonalizing the Hamiltonian matrix directly and separately for up and down spins.
        """

        solver_up = non_interacting_solver.EigenSolver(self.grids,
                                                       potential_fn=self.v_eff_up,
                                                       num_electrons=self.num_up_electrons,
                                                       boundary_condition=self.boundary_condition,
                                                       perturbation=self.fock_mat_up)
        solver_up.solve_ground_state()

        if self.num_down_electrons == 0:
            return self._update_ground_state(solver_up)
        else:
            solver_down = non_interacting_solver.EigenSolver(self.grids,
                                                             potential_fn=self.v_eff_down,
                                                             num_electrons=self.num_down_electrons,
                                                             boundary_condition=self.boundary_condition,
                                                             perturbation=self.fock_mat_down)
            solver_down.solve_ground_state()
            return self._update_ground_state(solver_up, solver_down)

    def solve_self_consistent_density(self, sym):

        delta_E = 1.0
        first_iter = True
        while delta_E > 1e-4:
            if not first_iter:
                old_E = self.E_tot

            # solve KS system -> obtain new new density
            self._solve_ground_state(first_iter, sym)

            # update total potentials using new density
            self._update_v_eff_up()
            self._update_v_eff_down()
            self._update_fock_matrix_up()
            self._update_fock_matrix_down()

            # Non-Interacting Kinetic Energy
            self.T_s = self.kinetic_energy

            # External Potential Energy
            self.V = (self.v_ext(self.grids) * self.density).sum() * self.dx

            v_h = self.hf.v_h()
            # Hartree Integral
            self.U = .5 * (v_h(grids=self.grids,
                               n=self.density) * self.density).sum() * self.dx

            # Exchange Energy
            self.E_x = self.get_E_x_HF()

            # Total Energy
            self.E_tot = self.T_s + self.V + self.U + self.E_x

            if not first_iter:
                delta_E = np.abs(old_E - self.E_tot).sum() * self.dx
            else:
                first_iter = False

        self._solved = True

        return self
