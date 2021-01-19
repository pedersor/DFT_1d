"""
SCF solver base
###############

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

    def __init__(self, grids, v_ext, num_electrons, num_unpaired_electrons,
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

        # Solver is not co nverged by default.
        self._converged = False
        self._init_spin_config(num_unpaired_electrons)


    def _init_spin_config(self, num_unpaired_electrons):
        """Default spin configuration. All unpaired electrons are defaulted to
        spin-up by convention.
        """

        if num_unpaired_electrons is None:
          return self
        elif (self.num_electrons - num_unpaired_electrons) % 2 != 0:
          raise ValueError('(num_electrons - num_unpaired_electrons) must be'
            'divisible by 2.')
        elif num_unpaired_electrons > self.num_electrons:
          raise ValueError('Cannot have num_unpaired_electrons > num_electrons')

        self.num_down_electrons = (self.num_electrons
                                   - num_unpaired_electrons) // 2
        self.num_up_electrons = self.num_down_electrons + num_unpaired_electrons
        return self

    def is_converged(self):
        """Returns whether the calculation has been converged."""
        return self._converged

    def _update_ground_state(self, solver_up, solver_down=None):
        """Helper function to _solve_ground_state() method.

        Updates the attributes total_energy, wave_function, density, kinetic_energy,
        potential_enenrgy and _solved from the eigensolver's output (w, v).
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

    def solve_self_consistent_density(self):
        """ Must be implemented in subclass. """
        return NotImplementedError()
