import numpy as np
np.set_printoptions(precision=15, suppress=True)

from kaufmann_solver.utils.utils import euclid_vector_norm
from tests.tests import hilb
from kaufmann_solver.bunch_kaufmann import symmetric_system_solve
from math import sqrt


mtx = hilb(10)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')

etalon = mtx.copy()

# We test solver for solve system Mx = M*(identity_vector)
original_solve = np.zeros(mtx.shape[0]) + 1
free_variables = mtx.dot(original_solve)

kaufmann_result = symmetric_system_solve(mtx, free_variables)

print '-'*80
print 'Kaufmann result:'
print kaufmann_result

print '-'*80
print 'Euclid norm of delta:'
print euclid_vector_norm(kaufmann_result - original_solve)
