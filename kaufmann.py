import numpy as np
from numpy import dot
np.set_printoptions(precision=15, suppress=True)

from tests.tests import hilb, factorization_test, linear_solve_test, numpy_test, extended_linear_solve_hilbert_test, cholesky_test, linear_cholesky

size = 10
mtx = hilb(20)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')
#mtx = np.random.randn(size, size)
#mtx = abs(mtx + mtx.T) / 2

"""mtx = np.random.randn(size, size)
chol = np.tril(mtx, -1)
np.fill_diagonal(chol, abs(np.random.randn(size)))
mtx = dot(chol, chol.T)"""

#flipper_test(mtx)
extended_linear_solve_hilbert_test(100)

#cholesky_test(mtx)
#linear_cholesky(mtx, True)
#print 'Chol test:'
#print chol
#factorization_test(mtx, False)
#linear_solve_test(mtx, precondition=False, regularize=False)
#linear_solve_test(mtx, precondition=True, regularize=False)
#numpy_test(mtx)