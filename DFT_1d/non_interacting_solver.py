# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Non-interacting Solver
######################

**Summary** 
    Solve non-interacting 1d system numerically on a grid
    (position-space basis). Each eigenstate will be occupied by one electron.

.. moduleauthor::
    EXAMPLE <Example@university.edu> <https://dft.uci.edu/> ORCID: `000-0000-0000-0000 <https://orcid.org/0000-0000-0000-0000>`_

.. note::
    Both solvers (EigenSolver, SparseEigenSolver) here are based on directly
    diagonalizing the Hamiltonian matrix.

.. todo::

    * Figure out what to do about joint copyright holders (Google + other)
    * Clean out unused example rst content in here.

"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from utils import get_dx, quadratic


class SolverBase:
    """Base Solver for non-interacting 1d system.

    Subclasses should define solve_ground_state method.
    """

    def __init__(self,
                 grids,
                 potential_fn=None,
                 num_electrons=1,
                 k_point=None,
                 boundary_condition='open',
                 n_point_stencil=3,
                 perturbation=None):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
              (num_grids,)
          potential_fn: potential function taking grids as argument.
          num_electrons: integer, the number of electrons in the system. Must be
              greater or equal to 1.
          k_point: the k-point in reciprocal space used to evaluate Hamiltonian
              for the case of a periodic potential. K should be chosen to be
              within the first Brillouin zone.
          boundary_condition:
              'closed': forward/backward finite difference methods will be used
              near the boundaries to ensure the wavefunction is zero at
              boundaries. This should only be used when the grid interval is
              purposefully small.

              'open': all ghost points outside of the grid are set to zero. This
              should be used whenever the grid interval is sufficiently large.
              Setting to false also results in a faster computational time due
              to matrix symmetry.

              'periodic': periodic (wrap-around) boundary conditions. Requires
              a k_point to be specified.
          n_point_stencil: total number of points used in the central finite
              difference method. The default 5-points results in a convergence
              rate of 4 for most systems. Suggested: use 3-point stencil for
              potentials with cusps as n_point_stencil > 3 will not improve
              convergence rates.
          perturbation: Provide a perturbation operator H' (a matrix) to append
              to the non-interacting Hamiltonian, H + H'.

        Raises:
          ValueError: If num_electrons is less than 1; or num_electrons is not
              an integer.
        """
        self.k = k_point
        self.boundary_condition = boundary_condition
        self.n_point_stencil = n_point_stencil
        self.grids = grids
        self.dx = get_dx(grids)
        self.num_grids = len(grids)
        self.potential_fn = potential_fn
        if self.potential_fn != None:
            self.vp = potential_fn(grids)
        self.perturbation = perturbation

        if not isinstance(num_electrons, int):
            raise ValueError('num_electrons is not an integer.')
        elif num_electrons < 1:
            raise ValueError(
                'num_electrons must be greater or equal to 1, but got %d'
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


class EigenSolver(SolverBase):
    """Represents the Hamiltonian as a matrix and diagonalizes it directly.

    This is the most stable and accurate eigensolver. Use SparseEigenSolver
    for a faster iterative eigensolver.
    """

    def __init__(self,
                 grids,
                 potential_fn=None,
                 num_electrons=1,
                 k_point=None,
                 boundary_condition='open',
                 n_point_stencil=3,
                 perturbation=None):
        """Initialize the solver with potential function and grid.

        Args:
          See SolverBase for args. discriptions.
        """
        super(EigenSolver, self).__init__(grids, potential_fn, num_electrons,
                                          k_point, boundary_condition,
                                          n_point_stencil, perturbation)
        self._set_matrices()

    def _diagonal_matrix(self, form):
        """Creates diagonal matrix.

        Attributes:
          form: string, creates identity matrix if form == 'identity'
                        creates potential matrix if form == 'potential'
        """
        if form == 'identity':
            return np.eye(self.num_grids, dtype=complex)
        elif form == 'potential':
            return np.diag(self.vp)

    def _set_matrices(self):
        """Sets matrix attributes.

        Attributes:
          _t_mat: numpy matrix, kinetic matrix in Hamiltonian.
          _v_mat: numpy matrix, potential matrix in Hamiltonian.
          _h: numpy matrix, Hamiltonian matrix.
        """

        # Kinetic matrix
        self._t_mat = self.get_kinetic_matrix()

        if self.potential_fn != None:
            # Potential matrix
            self._v_mat = self.get_potential_matrix()
            # Hamiltonian matrix
            self._h = self._t_mat + self._v_mat

        # Perturbation matrix
        if self.perturbation is not None:
            self._h += self.perturbation

    def update_potential(self, potential_fn):
        """Replace the current potential grids with a new potential grids.
        Delete all attributes created by solving eigenvalues, set _solved to
        False.
        
        Args:
          potential_fn: potential function taking grids as argument.
        """

        self.potential_fn = potential_fn
        self.vp = potential_fn(self.grids)
        # Potential matrix
        self._v_mat = self.get_potential_matrix()
        # Hamiltonian matrix
        self._h = self._t_mat + self._v_mat
        # Perturbation matrix
        if self.perturbation is not None:
            self._h += self.perturbation

        if self._solved:
            del self.total_energy
            del self.wave_function
            del self.density
            del self.kinetic_energy
            del self.potential_energy
            del self.eigenvalues
            self._solved = False

    def get_kinetic_matrix(self):
        """Kinetic matrix. Here the finite difference method is used to
        generate a kinetic energy operator in discrete space while satisfying
        desired boundary conditions.

        Returns:
          mat: Kinetic matrix.
            (num_grids, num_grids)
        """

        # n-point centered difference formula coefficients
        # these coefficients are for centered 2-order derivatives
        if self.n_point_stencil == 5:
            A_central = [-5 / 2, 4 / 3, -1 / 12]
        elif self.n_point_stencil == 3:
            A_central = [-2., 1.]
        else:
            raise ValueError(
                'n_point_stencil = %d is not supported' % self.n_point_stencil)

        mat = self._diagonal_matrix('identity')

        # get pentadiagonal KE matrix
        idx = np.arange(self.num_grids)
        for i, A_n in enumerate(A_central):
            mat[idx[i:], idx[i:] - i] = A_n
            mat[idx[:-i], idx[:-i] + i] = A_n

        # open-boundary
        if self.boundary_condition == 'open':
            mat = -0.5 * mat / (self.dx * self.dx)
            return np.real(mat)

        # append end-point forward/backward difference formulas
        elif self.boundary_condition == 'closed':

            if self.n_point_stencil == 5:
                # 0 means the first row, 1 means the second row
                # 0/1 are for forward/backward formulas at the end of the grid
                A_end_0 = [15 / 4, -77 / 6, 107 / 6, -13., 61 / 12, -5 / 6]
                A_end_1 = [5 / 6, -5 / 4, -1 / 3, 7 / 6, -1 / 2, 1 / 12]
            elif self.n_point_stencil == 3:
                # 0 means the same with 5 point
                A_end_0 = [2., -5., 4., -1.]

            # replace two ends of the matrix with forward/backward formulas
            for i, A_n_0 in enumerate(A_end_0):
                mat[0, i] = A_n_0
                mat[-1, -1 - i] = A_n_0
            if self.n_point_stencil == 5:
                for i, A_n_1 in enumerate(A_end_1):
                    mat[1, i] = A_n_1
                    mat[-2, -1 - i] = A_n_1

            # also change two end points as 0
            mat[0, 0] = 0
            mat[-1, -1] = 0

            mat = -0.5 * mat / (self.dx * self.dx)
            return np.real(mat)

        # periodic (no end point formulas needed)
        elif self.boundary_condition == 'periodic' and self.k is not None:
            k = self.k

            # assign central FDM formula (without center point)
            # also change 2nd-order end points
            if self.n_point_stencil == 3:
                D1_central = [1 / 2]

                mat[0, -1] = 1
                mat[-1, 0] = 1

            elif self.n_point_stencil == 5:
                D1_central = [2 / 3, -1 / 12]

                mat[0, -1] = 4 / 3
                mat[0, -2] = -1 / 12
                mat[1, -1] = -1 / 12

                mat[-1, 0] = 4 / 3
                mat[-1, 1] = -1 / 12
                mat[-2, 0] = -1 / 12

            # scale 2nd-order derivative matrix
            mat = -0.5 * mat / (self.dx * self.dx)

            # create identity matrix
            mat1 = 0.5 * (k ** 2) * self._diagonal_matrix('identity')

            # add 1st-order derivative matrix to identity
            idy = np.arange(self.num_grids)
            for i, D1_n in enumerate(D1_central):
                j = i + 1
                mat1[idy[0:], (idy[0:] - j) % self.num_grids] = \
                    complex(0., D1_n * k / self.dx)
                mat1[idy[0:], (idy[0:] + j) % self.num_grids] = \
                    complex(0., -D1_n * k / self.dx)

            # add all to second order matrix
            mat = mat + mat1

            return mat

        else:
            raise ValueError('boundary_condition = %s is not supported' %
                             self.boundary_condition)

    def get_potential_matrix(self):
        """Potential matrix. A diagonal matrix corresponding to the one-body
        potential input.

        Returns:
          mat: Potential matrix.
            (num_grids, num_grids)
        """

        if self.potential_fn == None:
            raise ValueError(
                'potential_fn is None, unable to get potential matrix.')

        return self._diagonal_matrix('potential')

    def _update_ground_state(self, eigenvalues, eigenvectors,
                             quadratic_function):
        """Helper function to solve_ground_state() method.

        Updates the attributes total_energy, wave_function, density,
        kinetic_energy, potential_enenrgy and _solved from the eigensolver's
        output (w, v).

        Args:
          eigenvalues: Numpy array with shape [num_eigenstates,], the
              eigenvalues in ascending order.
          eigenvectors: Numpy array with shape [num_grids, num_eigenstates],
              each column eigenvectors[:, i] is the normalized eigenvector
              corresponding to the eigenvalue eigenvalues[i].
          quadratic_function: Callable, compute the quadratic form of matrix and
              vector.

        Returns:
          self
        """
        self.total_energy = 0.
        self.wave_function = np.zeros((self.num_electrons, self.num_grids))
        self.density = np.zeros(self.num_grids)
        self.kinetic_energy = 0.
        self.potential_energy = 0.
        self.eigenvalues = eigenvalues

        for i in range(self.num_electrons):
            self.total_energy += eigenvalues[i]
            self.wave_function[i] = eigenvectors.T[i] / np.sqrt(self.dx)
            self.density += self.wave_function[i] ** 2
            self.kinetic_energy += quadratic_function(
                self._t_mat, self.wave_function[i]) * self.dx
            self.potential_energy += quadratic_function(
                self._v_mat, self.wave_function[i]) * self.dx

        self._solved = True
        return self

    def solve_ground_state(self):
        """Solve ground state by diagonalize the Hamiltonian matrix directly.

        Compute attributes:
        total_energy, kinetic_energy, potential_energy, density, wave_function.

        Returns:
          self
        """

        if self.potential_fn == None:
            raise ValueError(
                'potential_fn is None, unable to solve for ground state.')

        if (self.boundary_condition == 'open'
                or self.boundary_condition == 'periodic'):
            eigenvalues, eigenvectors = np.linalg.eigh(self._h)
        else:
            eigenvalues, eigenvectors = np.linalg.eig(self._h)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        return self._update_ground_state(eigenvalues, eigenvectors, quadratic)


class SparseEigenSolver(EigenSolver):
    """Represents the Hamiltonian as a matrix and solve with sparse eigensolver.

    This eigensolver is iterative and approximate. It is in general much
    faster than EigenSolver but less robust.
    """

    def __init__(self,
                 grids,
                 potential_fn=None,
                 num_electrons=1,
                 k_point=None,
                 boundary_condition='open',
                 n_point_stencil=3,
                 tol=10 ** -6):
        """Initialize the solver with potential function and grid.

        Args:
          tol: Relative accuracy for eigenvalues (stopping criterion).

          See SolverBase for additional args. discriptions.
        """
        super(SparseEigenSolver, self).__init__(grids, potential_fn,
                                                num_electrons, k_point,
                                                boundary_condition,
                                                n_point_stencil)
        self._tol = tol

    def _diagonal_matrix(self, form):
        """Creates diagonal matrix.

        Attributes:
          form: string, creates identity matrix if form == 'identity'
                        creates potential matrix if form == 'potential'
        """

        if form == 'identity':
            return sparse.eye(self.num_grids, dtype=complex, format="lil")
        elif form == 'potential':
            return sparse.diags(self.vp, offsets=0, format='lil')

    def _sparse_quadratic(self, sparse_matrix, vector):
        """Compute quadratic of a sparse matrix and a dense vector.

        As of Numpy 1.7, np.dot is not aware of sparse matrices, scipy suggests
        to use the matrix dot method: sparse_matrix.dot(vector).

        Args:
          sparse_matrix: Scipy sparse matrix with shape [dim, dim].
          vector: Numpy array with shape [dim].

        Returns:
          Float, quadratic form of the input matrix and vector.
        """
        return np.dot(vector, sparse_matrix.dot(vector))

    def solve_ground_state(self):
        """Solve ground state by sparse eigensolver.

        Compute attributes:
        total_energy, kinetic_energy, potential_energy, density, wave_function.

        Returns:
          self
        """
        if (self.boundary_condition == 'open'
                or self.boundary_condition == 'periodic'):
            eigenvalues, eigenvectors = linalg.eigsh(
                self._h, k=self.num_electrons,
                which='SA', tol=self._tol)
        else:
            eigenvalues, eigenvectors = linalg.eigs(
                self._h, k=self.num_electrons,
                which='SR', tol=self._tol)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        return self._update_ground_state(
            eigenvalues, eigenvectors, self._sparse_quadratic)
