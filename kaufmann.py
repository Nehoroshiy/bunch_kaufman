import numpy as np
np.set_printoptions(precision=15, suppress=True)

from tests.tests import hilb, factorization_test, linear_solve_test, numpy_test, extended_linear_solve_hilbert_test, flipper_test


mtx = hilb(15)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')

#flipper_test(mtx)
extended_linear_solve_hilbert_test(100)

#factorization_test(mtx)
#linear_solve_test(mtx)
#numpy_test(mtx)