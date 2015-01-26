import numpy as np
from numpy import dot
np.set_printoptions(precision=30, suppress=True, linewidth=250)

from tests.tests import hilb, factorization_test, linear_solve_test, numpy_test, extended_linear_solve_hilbert_test, cholesky_test, linear_cholesky

size = 10
max_size = 100
mtx = hilb(14)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')
#mtx = np.random.randn(size, size)
#mtx = abs(mtx + mtx.T) / 2

"""mtx = np.random.randn(size, size)
chol = np.tril(mtx, -1)
np.fill_diagonal(chol, abs(np.random.randn(size)))
mtx = dot(chol, chol.T)"""

#flipper_test(mtx)

"""for precondition in xrange(2):
    for trusty in xrange(2):
        for refinement in xrange(2):
            r = (refinement == 1)
            p = (precondition == 1)
            t = (trusty == 1)
            extended_linear_solve_hilbert_test(max_size, precond=p, trusty=t, refinement=r, dtype=np.float64)
            extended_linear_solve_hilbert_test(max_size, precond=p, trusty=t, refinement=r, dtype=np.float128)
"""
#cholesky_test(mtx)
#linear_cholesky(mtx, True)
#print 'Chol test:'
#print chol
#extended_linear_solve_hilbert_test(14, dtype=np.float128, trusty=True)
factorization_test(mtx, False)
#linear_solve_test(mtx, precondition=False, regularize=False)
#linear_solve_test(mtx, precondition=True, regularize=False)
#numpy_test(mtx)