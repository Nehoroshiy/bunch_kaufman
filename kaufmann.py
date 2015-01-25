import numpy as np
np.set_printoptions(precision=15, suppress=True)

from tests.tests import hilb, factorization_test, linear_solve_test, numpy_test, extended_linear_solve_hilbert_test, flipper_test

size = 20
mtx = hilb(29)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')
#mtx = np.random.randn(size, size)
#mtx = (mtx + mtx.T) / 1e5

#flipper_test(mtx)
#extended_linear_solve_hilbert_test(30)

factorization_test(mtx, False)
linear_solve_test(mtx, precondition=False, regularize=False)
linear_solve_test(mtx, precondition=True, regularize=False)
#numpy_test(mtx)