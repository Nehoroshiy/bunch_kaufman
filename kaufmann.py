import numpy as np
np.set_printoptions(precision=15, suppress=True)

from tests.tests import hilb, factorization_test, linear_solve_test


mtx = hilb(10)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')
etalon = mtx.copy()

factorization_test(mtx)
linear_solve_test(mtx)
