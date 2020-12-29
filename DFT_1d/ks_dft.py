"""
Kohn-Sham DFT solver
####################

**Summary** 
    Kohn-Sham DFT solver for 1-dimensional systems on a grid.

.. moduleauthor::
    `Ryan Pederson <pedersor@uci.edu>`_ ORCID: `0000-0002-7228-9478 <https://orcid.org/0000-0002-7228-9478>`_

.. todo::

    * Comments in KS solver funciton should be in doc format
    * *solve_self_consistent_density* needs summary sentence
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
        """Initialize the solver with an exchange-correlation (XC) functional.

        Args:
            xc: exchange correlation `functional` class object.
        """

        super(KS_Solver, self).__init__(grids, v_ext, num_electrons,
                                        boundary_condition)
        self.xc = xc
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

        self.v_s_up = functools.partial(self.xc.v_s_up,
                                        n=self.density, n_up=self.n_up,
                                        n_down=self.n_down, v_ext=self.v_ext,
                                        v_xc_up=self.xc.v_xc_up)
        return self

    def _update_v_s_down(self):
        """Total KS down spin potential to be solved self consistently in the
        KS system.
        """

        self.v_s_down = functools.partial(self.xc.v_s_down,
                                          n=self.density, n_up=self.n_up,
                                          n_down=self.n_down,
                                          v_ext=self.v_ext,
                                          v_xc_down=self.xc.v_xc_down)
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

    def solve_self_consistent_density(self, mixing_param=0.3, verbose=0):
        """Solve KS equations self-consistently.

        Args:
            mixing_param: linear mixing parameter, where 0.0 denotes no mixing.
            verbose: convergence debug printing.

        Returns:
            `KS_Solver`: converged `KS_Solver` with results.
        """

        # TODO: use prev_densities for DIIS mixing
        prev_densities = []

        final_energy = 1E100
        converged = False
        while not converged:
            # solve KS system -> obtain new density
            self._solve_ground_state()

            # update KS potential(s) using new density
            self._update_v_s()

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

        # Hartree Energy
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


class Spinless_KS_Solver(KS_Solver):
  """spinless KS-DFT solver for non-periodic systems."""

  def __init__(self, grids, v_ext, xc, num_electrons=1,
               boundary_condition='open'):
    """Initialize the solver with an exchange-correlation (XC) functional.

    Args:
        xc: (spinless) exchange correlation `functional` class object.
    """

    super(KS_Solver, self).__init__(grids, v_ext, num_electrons,
                                    boundary_condition)
    self.xc = xc
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

    self.v_s = functools.partial(self.xc.v_s,
                                 n=self.density, v_ext=self.v_ext,
                                 v_xc=self.xc.v_xc)
    return self

  def _solve_ground_state(self):
    """Solve ground state by diagonalizing the Hamiltonian matrix directly.
    """

    solver = non_interacting_solver.EigenSolver(
        self.grids,
        potential_fn=self.v_s,
        num_electrons=self.num_electrons,
        boundary_condition=self.boundary_condition)
    solver.solve_ground_state(occupation_per_state=2)

    return self._update_ground_state(solver)

  def _update_ground_state(self, solver):
    """Helper function to _solve_ground_state() method.

    Updates the attributes total_energy, wave_function, density, kinetic_energy,
    potential_enenrgy and _solved from the eigensolver's output (w, v).

    Overides _update_ground_state from `scf_base`.
    """

    self.kinetic_energy = solver.kinetic_energy
    self.eps = solver.kinetic_energy
    self.density = solver.density

    return self

  def solve_self_consistent_density(self, mixing_param=0.3, verbose=0):
    """Solve KS equations self-consistently.

    Args:
        mixing_param: linear mixing parameter, where 0.0 denotes no mixing.
        verbose: convergence debug printing.

    Returns:
        `KS_Solver`: converged `KS_Solver` with results.
    """

    # TODO: use prev_densities for DIIS mixing
    prev_densities = []

    final_energy = 1E100
    converged = False
    while not converged:
      # solve KS system -> obtain new density
      self._solve_ground_state()

      # update total potentials using new density
      self._update_v_s()

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

    # Hartree Energy
    v_h = self.xc.v_h()
    self.U = .5 * (v_h(grids=self.grids,
                       n=self.density) * self.density).sum() * self.dx

    # Exchange Energy
    self.E_x = self.xc.get_E_x(self.density)

    # Correlation Energy
    self.E_c = self.xc.get_E_c(self.density)

    # Total Energy
    self.E_tot = self.T_s + self.V + self.U + self.E_x + self.E_c

    return self


if __name__ == '__main__':
  import ks_dft, functionals, ext_potentials
  import numpy as np
  import functools

  h = 0.08
  grids = np.arange(-256, 257) * h
  num_electrons = 2
  nuclear_charge = 2

  v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=nuclear_charge)
  lda_xc = functionals.ExponentialLSDFunctional(grids=grids)
  solver = Spinless_KS_Solver(grids, v_ext=v_ext, xc=lda_xc,
                            num_electrons=num_electrons)
  solver.solve_self_consistent_density()

  # Non-Interacting (Kohn-Sham) Kinetic Energy
  print("T_s =", solver.T_s)

  # External Potential Energy
  print("V =", solver.V)

  # Hartree Energy
  print("U =", solver.U)

  # Exchange Energy
  print("E_x =", solver.E_x)

  # Correlation Energy
  print("E_c =", solver.E_c)

  # Total Energy
  print("E =", solver.E_tot)
