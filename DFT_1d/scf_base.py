"""
SCF solver base
####################

**Summary**
    This is the summary

.. moduleauthor::
    EXAMPLE <Example@university.edu> <https://dft.uci.edu/> ORCID: `000-0000-0000-0000 <https://orcid.org/0000-0000-0000-0000>`_

.. todo::

    * Authors?
    * *solve_self_consistent_density* needs summary sentence
    * Linting?
"""

from utils import get_dx


class SCF_SolverBase:
    """Base Solver for self-consistent field (SCF) calculations."""

    def __init__(self, grids, v_ext, num_electrons=1,
                 boundary_condition="open"):
        """Initialize the solver with external potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
              (num_grids,)
          v_ext: Kohn Sham external potential function taking grids as argument.
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
        """Returns whether the calculation has been converged."""
        return self._converged

    def solve_self_consistent_density(self):
        return NotImplementedError()
