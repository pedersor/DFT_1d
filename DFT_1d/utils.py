import numpy as np
import ext_potentials
import constants


class IntegralTool:
    '''Containing integral matrices.
    '''

    def __init__(self, grids, A=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
                 k=constants.EXPONENTIAL_COULOMB_KAPPA, init=True):
        self.grids = grids
        self.num_grids = len(grids)
        self.dx = np.abs(self.grids[-1] - self.grids[0]) / (self.num_grids - 1)
        self.A = A
        self.k = k
        if init:
            self.simpson_vector = self._get_simpson_vector()
            self.hartree_matrix = self._get_hartree_matrix()

    def integral(self, func_grids):
        if len(func_grids) != self.num_grids:
            raise ValueError(f"Input grids has dimension {func_grids}, "
                             f"it should have dimension {self.num_grids}.")

        return self.simpson_vector.dot(func_grids)

    def _simpson_coeff(self, index):
        if index == 0 or index == self.num_grids - 1:
            return 1
        elif index % 2 == 1:
            return 4
        else:
            return 2

    def _get_simpson_vector(self):
        v = np.empty(self.num_grids)
        for i in range(self.num_grids):
            v[i] = self._simpson_coeff(i) * self.dx / 3
        return v

    def _get_hartree_matrix(self):
        mat = np.empty([self.num_grids, self.num_grids])
        for i in range(self.num_grids):
            for j in range(self.num_grids):
                mat[i, j] = - self._simpson_coeff(j) * \
                            ext_potentials.exp_hydrogenic(
                                self.grids[i] - self.grids[j], self.A, self.k) * \
                            self.dx / 3
        return mat


class DerivativeTool:
    '''Containing derivative matrices.
    '''

    def __init__(self, grids, n_point_stencil=5, init=True):
        self.num_grids = len(grids)
        self.dx = (grids[-1] - grids[0]) / (self.num_grids - 1)
        self.n_point_stencil = n_point_stencil

        if init:
            self.d1_mat = self._get_d1_matrix()
            self.d2_mat = self._get_d2_matrix()

    def _get_d1_matrix(self):
        """get 1st order derivative matrix
        """

        # create identity matrix
        mat = np.eye(self.num_grids)

        if self.n_point_stencil == 5:

            # for centered 1st-order derivatives
            for i in range(2, self.num_grids - 2):
                mat[i][i - 2:i + 3] = [1 / 12, -2 / 3, 0., 2 / 3, -1 / 12]

            # 0 means the first (last) row, 1 means the second (second-last) row
            # 0 and 1 are for forward/backward formulas in two ends of the matrix
            end_0 = np.array([-25 / 12, 4., -3., 4 / 3, -1 / 4])
            end_1 = np.array([-1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12])
            # end_1 = np.array([0, -25/12, 4., -3., 4/3, -1/4]) #noend
            # end_0 = [0., 2/3, -1/12, 0, 0] #open
            # end_1 = [-2/3, 0., 2/3, -1/12, 0] #open
            # end_0 = [0, 0, 0, 0, 0] #zero
            # end_1 = [0, 0, 0, 0, 0] #zero

            mat[0][:5] = end_0
            mat[-1][-5:] = -end_0[::-1]
            mat[1][:5] = end_1
            mat[-2][-5:] = -end_1[::-1]
            # mat[1][:6] = end_1 #noend
            # mat[-2][-6:] = -end_1[::-1] #noend

        elif self.n_point_stencil == 3:
            for i in range(1, self.num_grids - 1):
                mat[i][i - 1:i + 2] = [-0.5, 0., 0.5]

            # 0 means the first (last) row
            end_0 = np.array([-3 / 2, 2., -1 / 2])
            # end_0 = [2., -1/2, 0] #open
            # end_0 = [0, 0, 0] #zero

            mat[0][:3] = end_0
            mat[-1][-3:] = -end_0[::-1]
        else:
            raise ValueError(
                'n_point_stencil = %d is not supported' % self.n_point_stencil)

        return mat / self.dx

    def _get_d2_matrix(self):
        """get 2nd order derivative matrix
        """

        # create identity matrix
        mat = np.eye(self.num_grids)

        if self.n_point_stencil == 5:

            # for centered 1st-order derivatives
            for i in range(2, self.num_grids - 2):
                mat[i][i - 2:i + 3] = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]

            # 0 means the first (last) row, 1 means the second (second-last) row
            # 0 and 1 are for forward/backward formulas in two ends of the matrix
            end_0 = [15 / 4, -77 / 6, 107 / 6, -13., 61 / 12, -5 / 6]
            end_1 = [5 / 6, -5 / 4, -1 / 3, 7 / 6, -1 / 2, 1 / 12]
            # end_0 = [35/12, -26/3, 19/2, -14/3, 11/12] #5
            # end_1 = [11/12, -5/3, 0.5, 1/3, -1/12] #5
            # end_0 = [-5/2, 4/3, -1/12, 0, 0, 0] #open
            # end_1 = [4/3, -5/2, 4/3, -1/12, 0, 0] #open
            # end_0 = [0, 0, 0, 0, 0, 0] #zero
            # end_1 = [0, 0, 0, 0, 0, 0] #zero

            mat[0][:6] = end_0
            mat[-1][-6:] = end_0[::-1]
            mat[1][:6] = end_1
            mat[-2][-6:] = end_1[::-1]

            # mat[0][:5] = end_0 #5
            # mat[-1][-5:] = end_0[::-1] #5
            # mat[1][:5] = end_1 #5
            # mat[-2][-5:] = end_1[::-1] #5

        elif self.n_point_stencil == 3:
            for i in range(1, self.num_grids - 1):
                mat[i][i - 1:i + 2] = [1., -2., 1.]

            # 0 means the first (last) row
            end_0 = [2., -5., 4., -1.]
            # end_0 = [-2., 1., 0, 0] #open
            # end_0 = [0, 0, 0, 0] #zero

            mat[0][:4] = end_0
            mat[-1][-4:] = end_0[::-1]
        else:
            raise ValueError(
                'n_point_stencil = %d is not supported' % self.n_point_stencil)

        return mat / (self.dx * self.dx)
