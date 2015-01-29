import numpy as np
from numpy import dot
np.set_printoptions(precision=30, suppress=True, linewidth=250)
from kaufmann_solver.bunch_kaufmann import bunch_kaufman
from tests.tests import hilb
#from tests.tests import time_test
import cProfile
from tests.tests import time_test, bunch_factorization_test, cholesky_factorization_test
#size = 10
#max_size = 100
#mtx = hilb(5, dtype=np.float128)
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
#linear_cholesky(mtx, False)
#print 'Chol test:'
#print chol

#extended_ckhol_linear_solve_random_test(300, step=5, dtype=np.float128, positive_def=True)
#print 'Cholesky ends'
#extended_ckhol_linear_solve_hilbert_test(100, step=1, dtype=np.float128)
#extended_linear_solve_hilbert_test(100, step=1, dtype=np.float128)
#fast_empiric_random_test(min_size=581, max_size=600, step=1, dtype=np.float128, positive_def=True)
#extended_double_linear_random_test(500, step=5, dtype=np.float128, positive_def=True)
#factorization_test(mtx)
#linear_solve_test(mtx, precondition=False, regularize=False)
#linear_solve_test(mtx, precondition=True, regularize=False)
#numpy_test(mtx)
#many_randoms(min_size=126, max_size=200, repeats=19, dtype=np.float128, positive_def=True)
#get_all_repeats(10, 200, 'bunch', 'random', np.float128)
#for i in xrange(2, 5, 1):
#    print 'file:', i
#    count_optimal_alpha(i, 50)
#time_test(10, 60)
cProfile.run('bunch_factorization_test(50)')
cProfile.run('cholesky_factorization_test(50)')
#bunch_kaufman(hilb(10))