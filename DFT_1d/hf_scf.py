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

        self.v_tot_up = v_ext
        self.v_tot_down = v_ext

        self.fock_mat_up = None
        self.fock_mat_down = None

        self.initialize_density()

    def initialize_density(self):
        # Get number of Up/Down Electrons. All unpaired electrons are defaulted to spin-up.

        num_UP_electrons = int(self.num_electrons / 2)
        num_DOWN_electrons = int(self.num_electrons / 2)
        if self.num_electrons % 2 == 1:
            num_UP_electrons += 1

        self.num_UP_electrons = num_UP_electrons
        self.num_DOWN_electrons = num_DOWN_electrons

        return self

    def update_v_tot_up(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_tot_up = functools.partial(self.hf.v_hf, n=self.density,
                                          v_ext=self.v_ext)
        return self

    def update_v_tot_down(self):
        # total potential to be solved self consistently in the Kohn Sham system

        self.v_tot_down = functools.partial(self.hf.v_hf, n=self.density,
                                            v_ext=self.v_ext)
        return self

    def update_fock_matrix_up(self):
        self.fock_mat_up = self.hf.update_fock_matrix(
            wave_function=self.wave_functionUP[:self.num_UP_electrons])

        return self

    def update_fock_matrix_down(self):
        if self.num_DOWN_electrons == 0:
            return self
        else:
            self.fock_mat_down = self.hf.update_fock_matrix(
                wave_function=self.wave_functionDOWN[:self.num_DOWN_electrons])

            return self

    def get_E_x_HF(self):
        if self.num_DOWN_electrons == 0:
            return self.hf.get_E_x(
                wave_function=self.wave_functionUP[:self.num_UP_electrons])
        else:
            E_x_up = self.hf.get_E_x(
                wave_function=self.wave_functionUP[:self.num_UP_electrons])
            E_x_down = self.hf.get_E_x(
                wave_function=self.wave_functionDOWN[:self.num_DOWN_electrons])
            return E_x_up + E_x_down

    def _update_ground_state(self, solverUP, first_iter, sym, solverDOWN=None):
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
            self.kinetic_energy += quadratic(solverUP._t_mat,
                                             solverUP.wave_function[
                                                 i]) * self.dx

        for i in range(self.num_DOWN_electrons):
            self.nDOWN += self.wave_functionDOWN[i] ** 2
            self.kinetic_energy += quadratic(solverDOWN._t_mat,
                                             solverDOWN.wave_function[
                                                 i]) * self.dx

        self.density = self.nUP + self.nDOWN
        self.zeta = (self.nUP - self.nDOWN) / (self.density)

        return self

    def solve_ground_state(self, first_iter, sym):
        """Solve ground state by diagonalizing the Hamiltonian matrix directly and separately for up and down spins.
        """

        solverUP = non_interacting_solver.EigenSolver(self.grids,
                                                      potential_fn=self.v_tot_up,
                                                      num_electrons=self.num_UP_electrons,
                                                      boundary_condition=self.boundary_condition,
                                                      perturbation=self.fock_mat_up)
        solverUP.solve_ground_state()

        if self.num_DOWN_electrons == 0:
            return self._update_ground_state(solverUP, first_iter, sym)
        else:
            solverDOWN = non_interacting_solver.EigenSolver(self.grids,
                                                            potential_fn=self.v_tot_down,
                                                            num_electrons=self.num_DOWN_electrons,
                                                            boundary_condition=self.boundary_condition,
                                                            perturbation=self.fock_mat_down)
            solverDOWN.solve_ground_state()
            return self._update_ground_state(solverUP, first_iter, sym,
                                             solverDOWN)

    def solve_self_consistent_density(self, sym):

        delta_E = 1.0
        first_iter = True
        while delta_E > 1e-4:
            if not first_iter:
                old_E = self.E_tot

            # solve KS system -> obtain new new density
            self.solve_ground_state(first_iter, sym)

            # update total potentials using new density
            self.update_v_tot_up()
            self.update_v_tot_down()
            self.update_fock_matrix_up()
            self.update_fock_matrix_down()

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
