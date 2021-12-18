"""
Hartree-Fock solver
###################

**Summary** 
    This is the summary

.. moduleauthor::
    `Ryan Pederson <pedersor@uci.edu>`_ ORCID: `0000-0002-7228-9478 <https://orcid.org/0000-0002-7228-9478>`_

.. todo::

    * Comments in HF solver funciton should be in doc format
    * *solve_self_consistent_density* needs summary sentence
    * Clean out unused example rst content in here.
    * linting?
"""

import functools

import numpy as np

from DFT_1d import non_interacting_solver
from DFT_1d.scf_base import SCF_SolverBase


class HF_Solver(SCF_SolverBase):
  """HF solver for non-periodic systems."""

  def __init__(self,
               grids,
               v_ext,
               hf,
               num_electrons,
               num_unpaired_electrons,
               boundary_condition='open'):
    """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
            (num_grids,)
          hf: HF class functional object.
          num_electrons: Integer, the number of electrons in the system.
        """
    super(HF_Solver, self).__init__(grids, v_ext, num_electrons,
                                    num_unpaired_electrons, boundary_condition)

    self.hf = hf(grids)
    self.init_v_eff()

  def init_v_eff(self,
                 v_eff_up=None,
                 v_eff_down=None,
                 fock_mat_up=None,
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
    """Total up potential to be solved self consistently in the Kohn Sham
        system."""

    self.v_eff_up = functools.partial(self.hf.v_hf,
                                      n=self.density,
                                      v_ext=self.v_ext)
    return self

  def _update_v_eff_down(self):
    """Total up potential to be solved self consistently in the Kohn Sham
        system."""

    self.v_eff_down = functools.partial(self.hf.v_hf,
                                        n=self.density,
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

  def get_hf_exchange_energy(self):
    if self.num_down_electrons == 0:
      return self.hf.get_exchange_energy(
          wave_function=self.phi_up[:self.num_up_electrons])
    else:
      E_x_up = self.hf.get_exchange_energy(
          wave_function=self.phi_up[:self.num_up_electrons])
      E_x_down = self.hf.get_exchange_energy(
          wave_function=self.phi_down[:self.num_down_electrons])
      return E_x_up + E_x_down

  def _solve_ground_state(self):
    """Solve ground state by diagonalizing the Hamiltonian matrix directly and separately for up and down spins.
        """

    solver_up = non_interacting_solver.EigenSolver(
        self.grids,
        potential_fn=self.v_eff_up,
        num_electrons=self.num_up_electrons,
        boundary_condition=self.boundary_condition,
        perturbation=self.fock_mat_up)
    solver_up.solve_ground_state()

    if self.num_down_electrons == 0:
      return self._update_ground_state(solver_up)
    else:
      solver_down = non_interacting_solver.EigenSolver(
          self.grids,
          potential_fn=self.v_eff_down,
          num_electrons=self.num_down_electrons,
          boundary_condition=self.boundary_condition,
          perturbation=self.fock_mat_down)
      solver_down.solve_ground_state()
      return self._update_ground_state(solver_up, solver_down)

  def solve_self_consistent_density(self,
                                    energy_converge_tolerance=1e-6,
                                    mixing_param=0.3,
                                    max_iterations=50,
                                    verbose=0):
    """Solve HF equations self-consistently.

        Args:
            mixing_param: linear mixing parameter, where 0.0 denotes no mixing.
            verbose: convergence debug printing.

        Returns:
            `KS_Solver`: converged/non-converged `KS_Solver` with results.
        """
    # TODO: use prev_densities for DIIS mixing
    prev_densities = []

    previous_energy = None
    for i in range(max_iterations):
      # solve HF eqs. -> obtain new new density
      self._solve_ground_state()

      # Non-Interacting Kinetic Energy
      self.hf_kinetic_energy = self.kinetic_energy

      # External Potential Energy
      self.ext_potential_energy = (self.v_ext(self.grids) *
                                   self.density).sum() * self.dx

      # Hartree Energy
      hartree_potential = self.hf.hartree_potential()
      self.hartree_energy = .5 * (hartree_potential(
          grids=self.grids, n=self.density) * self.density).sum() * self.dx

      # Exchange Energy
      self.exchange_energy = self.get_hf_exchange_energy()

      # Total Energy
      self.total_energy = (self.hf_kinetic_energy + self.ext_potential_energy +
                           self.hartree_energy + self.exchange_energy)

      if previous_energy is None:
        pass
      elif (np.abs(self.total_energy - previous_energy) <
            energy_converge_tolerance):
        self._converged = True
        break
      elif prev_densities and mixing_param:
        self.n_up = ((1 - mixing_param) * self.n_up +
                     mixing_param * prev_densities[-1][0])
        self.n_down = ((1 - mixing_param) * self.n_down +
                       mixing_param * prev_densities[-1][1])
        self.density = self.n_up + self.n_down

      # update eff potentials using new density
      self._update_v_eff_up()
      self._update_v_eff_down()

      # update fock matrix using new orbitals {phi}
      self._update_fock_matrix_up()
      self._update_fock_matrix_down()

      previous_energy = self.total_energy
      prev_densities.append((self.n_up, self.n_down))

      # TODO: add more verbose options
      if verbose == 1:
        print(f"i = {i}: E = {previous_energy}")

    return self
