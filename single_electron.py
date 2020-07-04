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

"""Solvers for non-interacting 1d system.

Solve non-interacting 1d system numerically on grids. Each eigenstate will be
occupied by one electron.

Note both solver (EigenSolver, SparseEigenSolver) here are based on directly
diagonalizing the Hamiltonian matrix, which are straightforward to understand,
but not as accurate as other delicate numerical methods, like density matrix
renormalization group (DMRG).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import sparse
from scipy.sparse import linalg


# import tensorflow as tf


def get_dx(grids):
    """Gets the grid spacing from grids array.

    Args:
      grids: Numpy array with shape (num_grids,).

    Returns:
      Grid spacing.
    """
    return (grids[-1] - grids[0]) / (len(grids) - 1)


def vw_grid(density, dx):
    """von Weizsacker kinetic energy functional on grid.

    Args:
      density: numpy array, density on grid.
        (num_grids,)
      dx: grid spacing.

    Returns:
      kinetic_energy: von Weizsacker kinetic energy.
    """
    gradient = np.gradient(density) / dx
    return np.sum(0.125 * gradient * gradient / density) * dx


def quadratic(mat, x):
    """Compute the quadratic form of matrix and vector.

    Args:
      mat: matrix.
        (n, n)
      x: vector.
        (n,)

    Returns:
      output: scalar value as result of x A x.T.
    """
    return np.dot(x, np.dot(mat, x))


class SolverBase(object):
    """Base Solver for non-interacting 1d system.

    Subclasses should define solve_ground_state method.
    """

    def __init__(self, grids, potential_fn, num_electrons=1, k_point=None,
                 boundary_condition='open',
                 n_point_stencil=5, fock_mat=None):
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
          fock_mat: Provide a Fock operator (matrix) if using hartree-fock
              Hamiltonian

        Raises:
          ValueError: If num_electrons is less than 1; or num_electrons is not
              an integer.
        """
        # 1d grids.
        self.k = k_point
        self.boundary_condition = boundary_condition
        self.n_point_stencil = n_point_stencil
        self.grids = grids
        self.dx = get_dx(grids)
        self.num_grids = len(grids)
        # Potential on grid.
        self.vp = potential_fn(grids)
        self.fock_mat = fock_mat

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
    """

    def __init__(self, grids, potential_fn, num_electrons=1, k_point=None,
                 boundary_condition='open',
                 n_point_stencil=5, fock_mat=None):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
            (num_grids,)
          potential_fn: potential function taking grids as argument.
          num_electrons: Integer, the number of electrons in the system.
        """
        super(EigenSolver, self).__init__(grids, potential_fn, num_electrons,
                                          k_point, boundary_condition,
                                          n_point_stencil, fock_mat)
        self._set_matrices()

    def _set_matrices(self):
        """Sets matrix attributes.

        Attributes:
          _t_mat: Scipy sparse matrix, kinetic matrix in Hamiltonian.
          _v_mat: Scipy sparse matrix, potential matrix in Hamiltonian.
          _h: Scipy sparse matrix, Hamiltonian matrix.
        """
        # Kinetic matrix
        self._t_mat = self.get_kinetic_matrix()
        # Potential matrix
        self._v_mat = self.get_potential_matrix()
        # Hamiltonian matrix
        self._h = self._t_mat + self._v_mat
        # Fock-Matrix (exact exchange)
        if self.fock_mat is not None:
            self._h += self.fock_mat

    def get_kinetic_matrix(self):
        """Kinetic matrix. Here the finite difference method is used to
        generate a kinetic energy operator in discrete space while satisfying
        desired boundary conditions.

        Returns:
          mat: Kinetic matrix.
            (num_grids, num_grids)
        """

        # n-point centered difference formula coefficients
        if self.n_point_stencil == 5:
            A_central = [-5 / 2, 4 / 3, -1 / 12]
        elif self.n_point_stencil == 3:
            A_central = [-2., 1.]
        else:
            raise ValueError(
                'n_point_stencil = %s is not supported' % self.n_point_stencil)

        mat = A_central[0] * sparse.eye(self.num_grids, format="lil")
        for i, A_n in enumerate(A_central[1:]):
            j = i + 1
            elements = A_n * np.ones(self.num_grids - j)
            mat += sparse.diags(elements, offsets=j, format="lil")
            mat += sparse.diags(elements, offsets=-j, format="lil")
    
        if (self.boundary_condition == 'open'):
            mat = -0.5 * mat / (self.dx * self.dx)
            return mat
        # end-point forward/backward difference formulas
        elif (self.boundary_condition == 'closed'):
            
            if self.n_point_stencil == 5:
                # 0 means the first row, 1 means the second row
                # 0 and 1 are for forward/backward formulas in two ends of the matrix
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
            return mat

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
                mat[-1, -2] = -1 / 12
                mat[-2, -2] = -1 / 12
            
            # scale 2nd-order derivative matrix
            mat = -0.5 * mat / (self.dx * self.dx)
            
            # create identity matrix
            mat1 = 0.5 * (k ** 2) * sparse.eye(self.num_grids, dtype=complex, format="lil")
            
            # add 1st-order derivative matrix to identity
            idy = np.arange(self.num_grids)       
            for i, D1_n in enumerate(D1_central):
                j = i + 1
                mat1[idy[0:], (idy[0:] - j) % self.num_grids] = complex(0., D1_n * k / self.dx)
                mat1[idy[0:], (idy[0:] + j) % self.num_grids] = complex(0., -D1_n * k / self.dx)
            
            
            # add all to second order matrix
            mat = mat + mat1
            
            return mat
        else:
            raise ValueError(
                'boundary_condition = %s is not supported' % self.boundary_condition)

    def get_potential_matrix(self):
        """Potential matrix. A diagonal matrix corresponding to the one-body
        potential input.

        Returns:
          mat: Potential matrix.
            (num_grids, num_grids)
        """

        return np.diag(self.vp)

    def _update_ground_state(self, eigenvalues, eigenvectors,
                             quadratic_function):
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
        if (self.boundary_condition == 'open'
                or self.boundary_condition == 'periodic'):
            eigenvalues, eigenvectors = np.linalg.eigh(self._h)
        else:
            eigenvalues, eigenvectors = np.linalg.eig(self._h)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        return self._update_ground_state(eigenvalues, eigenvectors, quadratic)


# TODO(Chris): repeat everything for sparse matrices
class SparseEigenSolver(EigenSolver):
    """Represents the Hamiltonian as a matrix and solve with sparse eigensolver.
    """

    def __init__(self,
                 grids,
                 potential_fn,
                 num_electrons=1,
                 additional_levels=5, k_point=None, boundary_condition=False,
                 n_point_stencil=5):
        """Initialize the solver with potential function and grid.

        Args:
          grids: numpy array of grid points for evaluating 1d potential.
            (num_grids,)
          potential_fn: potential function taking grids as argument.
          num_electrons: Integer, the number of electrons in the system.
          additional_levels: Integer, non-negative number. For numerical accuracy of
            eigen energies for the first num_electrons,
            num_electrons + additional_levels will be solved.

        Raises:
          ValueError: If additional_levels is negative.
        """
        super(SparseEigenSolver, self).__init__(grids, potential_fn,
                                                num_electrons, k_point,
                                                boundary_condition,
                                                n_point_stencil)
        if additional_levels < 0:
            raise ValueError(
                'additional_levels is expected to be non-negative, but '
                'got %d.' % additional_levels)
        elif additional_levels > self.num_grids - self.num_electrons:
            raise ValueError('additional_levels is expected to be smaller than '
                             'num_grids - num_electrons (%d), but got %d.'
                             % (self.num_grids - self.num_electrons,
                                additional_levels))
        self._additional_levels = additional_levels
        self._set_matrices()

    
    def get_kinetic_matrix(self):
        """Kinetic matrix. Here the finite difference method is used to
        generate a kinetic energy operator in discrete space while satisfying
        desired boundary conditions.

        Returns:
          mat: Kinetic matrix.
            (num_grids, num_grids)
        """

        # n-point centered difference formula coefficients
        if self.n_point_stencil == 5:
            A_central = [-5 / 2, 4 / 3, -1 / 12]
        elif self.n_point_stencil == 3:
            A_central = [-2., 1.]
        else:
            raise ValueError(
                'n_point_stencil = %s is not supported' % self.n_point_stencil)

        mat = A_central[0] * sparse.eye(self.num_grids, format="lil")
        for i, A_n in enumerate(A_central[1:]):
            j = i + 1
            elements = A_n * np.ones(self.num_grids - j)
            mat += sparse.diags(elements, offsets=j, format="lil")
            mat += sparse.diags(elements, offsets=-j, format="lil")
    
        if (self.boundary_condition == 'open'):
            mat = -0.5 * mat / (self.dx * self.dx)
            return mat
        # end-point forward/backward difference formulas
        elif (self.boundary_condition == 'closed'):
            
            if self.n_point_stencil == 5:
                # 0 means the first row, 1 means the second row
                # 0 and 1 are for forward/backward formulas in two ends of the matrix
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
            return mat

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
                mat[-1, -2] = -1 / 12
                mat[-2, -2] = -1 / 12
            
            # scale 2nd-order derivative matrix
            mat = -0.5 * mat / (self.dx * self.dx)
            
            # create identity matrix
            mat1 = 0.5 * (k ** 2) * sparse.eye(self.num_grids, dtype=complex, format="lil")
            
            # add 1st-order derivative matrix to identity
            idy = np.arange(self.num_grids)       
            for i, D1_n in enumerate(D1_central):
                j = i + 1
                mat1[idy[0:], (idy[0:] - j) % self.num_grids] = complex(0., D1_n * k / self.dx)
                mat1[idy[0:], (idy[0:] + j) % self.num_grids] = complex(0., -D1_n * k / self.dx)
            
            
            # add all to second order matrix
            mat = mat + mat1
            
            return mat
        
        else:
            raise ValueError(
                'boundary_condition = %s is not supported' % self.boundary_condition)
            

    def get_potential_matrix(self):
        """Potential matrix.

        Returns:
          mat: Potential matrix.
            (num_grids, num_grids)
        """
        return sparse.diags(self.vp, offsets=0)

    def _sparse_quadratic(self, sparse_matrix, vector):
        """Compute quadratic of a sparse matrix and a dense vector.

        As of Numpy 1.7, np.dot is not aware of sparse matrices, scipy suggests to
        use the matrix dot method: sparse_matrix.dot(vector).

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
        # NOTE(leeley): linalg.eigsh is built on ARPACK. ArpackNoConvergence will be
        # raised if convergence is not obtained.
        # eigsh will solve 5 more eigenstates than self.num_electrons to reduce the
        # numerical error for the last few eigenstates.

        if (self.boundary_condition == 'open'):
            eigenvalues, eigenvectors = linalg.eigsh(
                self._h, k=self.num_electrons + self._additional_levels,
                which='SM')
        else:
            eigenvalues, eigenvectors = linalg.eigs(
                self._h, k=self.num_electrons + self._additional_levels,
                which='SM')
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        return self._update_ground_state(
            eigenvalues, eigenvectors, self._sparse_quadratic)


def solved_1dsolver_to_example(solver, params):
    """Converts an solved solver with a name to a tf.Example proto.

    Args:
      solver: A Solver instance with attribute solved=True.
      params: dict, other parameters to store in the tf.Example proto.

    Returns:
      example: A tf.Example proto with the following populated fields:
        density, kinetic_energy, total_energy, potential, and other keys in params
        dict.

    Raises:
      ValueError: If the solver is not solved.
    """
    if not solver.is_solved():
        raise ValueError('Input solver is not solved.')

    example = tf.train.Example()
    example.features.feature['density'].float_list.value.extend(
        list(solver.density))
    example.features.feature['kinetic_energy'].float_list.value.append(
        solver.kinetic_energy)
    example.features.feature['total_energy'].float_list.value.append(
        solver.total_energy)
    example.features.feature['dx'].float_list.value.append(
        solver.dx)
    example.features.feature['potential'].float_list.value.extend(
        list(solver.vp))
    for key, value in six.iteritems(params):
        if isinstance(value, (list, np.ndarray)):
            example.features.feature[key].float_list.value.extend(list(value))
        else:
            example.features.feature[key].float_list.value.append(value)

    return example
