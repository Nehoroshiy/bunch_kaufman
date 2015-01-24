import numpy as np
np.set_printoptions(precision=15, suppress=True)

from tests.tests import hilb, factorization_test, linear_solve_test, numpy_test, extended_factorization_test


mtx = hilb(14)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')
etalon = mtx.copy()

#extended_factorization_test(mtx)

factorization_test(mtx)
linear_solve_test(mtx)
numpy_test(mtx)