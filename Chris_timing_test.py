import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import six
from six.moves import range
import copy
import timeit

from single_electron import *

#print(np.diag(np.full(1, 1), 2))

#tester = EigenSolver(np.array([i for i in range(12)]), lambda x:x^2, boundary_condition = "exponential decay", n_point_stencil=3)
#print(np.around(tester.get_kinetic_matrix()*(tester.dx**2)/(-0.5), 1))

#==================variables======================
num_grids = 10000
A_central = [-5 / 2, 4 / 3, -1 / 12]
A_end_0 = [15 / 4, -77 / 6, 107 / 6, -13., 61 / 12, -5 / 6]
A_end_1 = [6 / 5, -5 / 4, -1 / 3, 7 / 6, -1 / 2, 1 / 12]
#=================================================

#====================codes1========================
mat1 = np.eye(num_grids)
idx = np.arange(num_grids)

#pentadiagonal
for j, A_n in enumerate(A_central):
    mat1[idx[j:], idx[j:] - j] = A_n
    mat1[idx[:-j], idx[:-j] + j] = A_n

start_time = timeit.default_timer()

#ends
for i, A_n_0 in enumerate(A_end_0):
    mat1[0, i] = A_n_0
    mat1[-1, -1 - i] = A_n_0
for i, A_n_1 in enumerate(A_end_1):
    mat1[1, i] = A_n_1
    mat1[-2, -1 - i] = A_n_1
    
elapsed = timeit.default_timer() - start_time
#=================================================

print(elapsed)


#====================codes2========================
mat2 = np.eye(num_grids)
idx = np.arange(num_grids)

#pentadiagonal
for j, A_n in enumerate(A_central):
    mat2[idx[j:], idx[j:] - j] = A_n
    mat2[idx[:-j], idx[:-j] + j] = A_n

start_time = timeit.default_timer()

#ends
mat2[0, :6] = A_end_0
mat2[-1, -6:] = A_end_0[::-1]
mat2[1, :6] = A_end_1
mat2[-2, -6:] = A_end_1[::-1]

elapsed = timeit.default_timer() - start_time
#=================================================

print(elapsed)
print(np.array_equal(mat1, mat2))