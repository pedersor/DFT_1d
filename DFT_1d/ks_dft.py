"""
Kohn-Sham DFT solver
####################

**Summary** 
    Kohn-Sham DFT solver for 1-dimensional systems on a grid.

.. moduleauthor::
    `Ryan Pederson <pedersor@uci.edu>`_ ORCID: `0000-0002-7228-9478 <https://orcid.org/0000-0002-7228-9478>`_

.. todo::

    * Linting?
"""

import functools

import numpy as np

from DFT_1d import non_interacting_solver, functionals
from DFT_1d.utils import quadratic
from DFT_1d.scf_base import SCF_SolverBase


class KS_Solver(SCF_SolverBase):
  """KS-DFT solver for non-periodic systems."""

  def __init__(self,
               grids,
               v_ext,
               xc,
               num_electrons,
               num_unpaired_electrons,
               occupation_per_state=2,
               boundary_condition='open'):
    """Initialize the solver with an exchange-correlation (XC) functional.

        Args:
            xc: exchange correlation `functional` class object.
        """

    super(KS_Solver,
          self).__init__(grids, v_ext, num_electrons, num_unpaired_electrons,
                         occupation_per_state, boundary_condition)
    self.xc = xc(grids)
    self.init_v_s()

  def init_v_s(self, v_s_up=None, v_s_down=None):
    """Initialize starting v_s_up and v_s_down. The default
        corresponds to v_hxc_up = v_hxc_down = 0.

        Args:
            v_s_up: initial v_s_up on a grid.
            v_s_down: initial v_s_down on a grid.
        """

    if v_s_up is None and v_s_down is None:
      # default initalization, v_s = v_ext.
      self.v_s_up = self.v_ext
      self.v_s_down = self.v_ext
    else:
      self.v_s_up = v_s_up
      self.v_s_down = v_s_down
    return self

  def _update_v_s_up(self):
    """Total KS up spin potential to be solved self consistently in the
        KS system.
        """

    self.v_s_up = functools.partial(self.xc.get_ks_potential_up,
                                    n_up=self.n_up,
                                    n_down=self.n_down,
                                    v_ext=self.v_ext)
    return self

  def _update_v_s_down(self):
    """Total KS down spin potential to be solved self consistently in the
        KS system.
        """

    self.v_s_down = functools.partial(self.xc.get_ks_potential_down,
                                      n_up=self.n_up,
                                      n_down=self.n_down,
                                      v_ext=self.v_ext)
    return self

  def _update_v_s(self):
    """Update KS potential(s)."""

    self._update_v_s_up()
    self._update_v_s_down()
    return self

  def _solve_ground_state(self):
    """Solve ground state by diagonalizing the Hamiltonian matrix directly
        and separately for up and down spins.
        """

    solver_up = non_interacting_solver.EigenSolver(
        self.grids,
        potential_fn=self.v_s_up,
        num_electrons=self.num_up_electrons,
        boundary_condition=self.boundary_condition)
    solver_up.solve_ground_state()

    if self.num_down_electrons == 0:
      return self._update_ground_state(solver_up)
    else:
      solver_down = non_interacting_solver.EigenSolver(
          self.grids,
          potential_fn=self.v_s_down,
          num_electrons=self.num_down_electrons,
          boundary_condition=self.boundary_condition)
      solver_down.solve_ground_state()
      return self._update_ground_state(solver_up, solver_down)

  def solve_self_consistent_density(self,
                                    energy_converge_tolerance=1e-6,
                                    mixing_param=0.3,
                                    max_iterations=50,
                                    verbose=0):
    """Solve KS equations self-consistently.

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
      # solve KS system -> obtain new density
      self._solve_ground_state()

      # Non-interacting (Kohn-Shame) kinetic energy
      self.ks_kinetic_energy = self.kinetic_energy

      # External potential energy
      self.ext_potential_energy = (self.v_ext(self.grids) *
                                   self.density).sum() * self.dx

      # Hartree energy
      self.hartree_energy = self.xc.get_hartree_energy(self.density)

      # Exchange energy
      self.exchange_energy = self.xc.get_exchange_energy(self.n_up, self.n_down)

      # Correlation energy
      self.correlation_energy = self.xc.get_correlation_energy(
          self.n_up, self.n_down)

      # Total energy
      self.total_energy = (self.ks_kinetic_energy + self.ext_potential_energy +
                           self.hartree_energy + self.exchange_energy +
                           self.correlation_energy)

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

      self._update_v_s()

      previous_energy = self.total_energy
      prev_densities.append((self.n_up, self.n_down))

      # TODO: add more verbose options
      if verbose == 1:
        print(f"i = {i}: E = {previous_energy}", flush=True)

    return self


class Spinless_KS_Solver(KS_Solver):
  """spinless KS-DFT solver for non-periodic systems."""

  def __init__(self,
               grids,
               v_ext,
               xc,
               num_electrons,
               occupation_per_state=2,
               boundary_condition='open'):
    """Initialize the solver with an exchange-correlation (XC) functional.

    Args:
        xc: (spinless) exchange correlation `functional` class object.
    """

    super(KS_Solver, self).__init__(grids,
                                    v_ext,
                                    num_electrons,
                                    num_unpaired_electrons=None,
                                    occupation_per_state=occupation_per_state,
                                    boundary_condition=boundary_condition)
    self.xc = xc(grids)
    self.init_v_s()

  def init_v_s(self, v_s=None):
    """Initialize starting v_s. The default
    corresponds to v_hxc = 0.

    Args:
        v_s: initial v_s on a grid.
    """

    if v_s is None:
      # default initalization, v_s = v_ext.
      self.v_s = self.v_ext
    else:
      self.v_s = v_s
    return self

  def _update_v_s(self):
    """Total KS potential to be solved self consistently in the
    KS system.
    """

    self.v_s = functools.partial(self.xc.get_ks_potential,
                                 n=self.density,
                                 v_ext=self.v_ext)
    return self

  def _solve_ground_state(self):
    """Solve ground state by diagonalizing the Hamiltonian matrix directly.
    """

    solver = non_interacting_solver.EigenSolver(
        self.grids,
        potential_fn=self.v_s,
        num_electrons=self.num_electrons,
        boundary_condition=self.boundary_condition)
    solver.solve_ground_state(occupation_per_state=self.occupation_per_state)

    return self._update_ground_state(solver)

  def _update_ground_state(self, solver):
    """Helper function to _solve_ground_state() method.

    Updates the attributes total_energy, wave_function, density, kinetic_energy,
    potential_enenrgy and _solved from the eigensolver's output (w, v).

    Overides _update_ground_state from `scf_base`.
    """

    self.kinetic_energy = solver.kinetic_energy
    self.eps = solver.total_energy
    self.density = solver.density
    self.phi = solver.wave_function

    return self

  def solve_self_consistent_density(self,
                                    energy_converge_tolerance=1e-6,
                                    mixing_param=0.3,
                                    max_iterations=50,
                                    verbose=0):
    """Solve KS equations self-consistently.

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
      # solve KS system -> obtain new density
      self._solve_ground_state()

      # Non-interacting (Kohn-Shame) kinetic energy
      self.ks_kinetic_energy = self.kinetic_energy

      # External potential energy
      self.ext_potential_energy = (self.v_ext(self.grids) *
                                   self.density).sum() * self.dx

      # Hartree energy
      self.hartree_energy = self.xc.get_hartree_energy(self.density)

      # Exchange energy
      self.exchange_energy = self.xc.get_exchange_energy(self.density)

      # Correlation energy
      self.correlation_energy = self.xc.get_correlation_energy(self.density)

      # Total energy
      self.total_energy = (self.ks_kinetic_energy + self.ext_potential_energy +
                           self.hartree_energy + self.exchange_energy +
                           self.correlation_energy)

      if previous_energy is None:
        pass
      elif (np.abs(self.total_energy - previous_energy) <
            energy_converge_tolerance):
        self._converged = True
        break
      elif prev_densities and mixing_param:
        self.density = ((1 - mixing_param) * self.density +
                        mixing_param * prev_densities[-1])

      # update KS potential(s) using new density
      self._update_v_s()

      previous_energy = self.total_energy
      prev_densities.append(self.density)

      # TODO: add more verbose options
      if verbose == 1:
        print(f"i = {i}: E = {previous_energy}", flush=True)

    return self
