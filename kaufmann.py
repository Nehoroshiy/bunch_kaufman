import numpy as np
np.set_printoptions(precision=15, suppress=True)

from kaufmann_solver.utils.utils import frobenius_norm, relative_error
from tests.algorithm_tuner import vizualize_bunch_kaufman_tune, get_optimal_alpha
from tests.tests import hilb
from tests.algorithm_tuner import hilbert_matrix_test
from kaufmann_solver.bunch_kaufmann import bunch_kaufmann, symmetric_system_solve
from math import sqrt



mtx = hilb(20)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')

etalon = mtx.copy()

open("result.txt", "w").write(str(mtx))

stripped_matrix, P, L, cell_sizes = bunch_kaufmann(mtx, (1.0 + sqrt(17.0))/8)

print ", ".join(map(lambda t: str(t), stripped_matrix.diagonal()))
print 'Matrix:'
print etalon
print '-'*80
print 'Check:'
result = P.dot(L).dot(stripped_matrix).dot(np.matrix(P.dot(L)).getH())
print result
print cell_sizes
print '-'*80

print '-'*80
print 'Resulted matrix'
print stripped_matrix
print stripped_matrix.nonzero()
print '-'*80
print 'And PL matrix:'
print P.dot(L)


print etalon.shape
print result.shape

print frobenius_norm(etalon - result)

#print frobenius_norm(etalon - np.random.random([101, 101]))

"""
print '\n'*7
print '-'*80

mtx = etalon.copy()
free_variables = mtx.dot(np.zeros(mtx.shape[0]) + 1)
print '-'*80
print 'matrix:'
print mtx
print 'Determinant of matrix'
print np.linalg.det(mtx)
print 'Vector of free variables'
print free_variables
"""
#kaufmann_result = symmetric_system_solve(mtx, free_variables)
#print 'Kaufmann linear solver:'
#print mtx.dot(kaufmann_result), free_variables
#print 'frobenius norm of difference:'
#print kaufmann_result.shape, free_variables.shape
#print sqrt(sum([(first - second)**2 for (first, second) in zip(kaufmann_result, free_variables)]))
#print '-'*80
#print '\n'

#ordinary_result = np.linalg.solve(mtx, free_variables)
#print 'Ordinary linear solver:'
#print mtx.dot(ordinary_result), free_variables
#print 'frobenius norm of difference:'
#print sqrt(sum([(first - second)**2 for (first, second) in zip(ordinary_result, free_variables)]))
#print frobenius_norm(kaufmann_result - ordinary_result)

#mtx = hilb(8)
#print '\n'*5
#print '-'*80
#print 'matrix:'
#print mtx
#print '-'*80
#print 'linear system solver:'
#print

#k = 39
#hb = hilb(k)
#original_vector = np.zeros(k) + 1
#print relative_error(original_vector, symmetric_system_solve(hb, np.dot(hb, original_vector)))
#alpha_optimal = get_optimal_alpha(hb, original_vector, 0.0, 1.0, 0.5)
#print alpha_optimal, (1.0 + sqrt(17))/8

#vizualize_bunch_kaufman_tune(hilb(50), 5)
#hilbert_matrix_test(100)

"""for i in xrange(5, 100, 1):
    original_result = np.zeros(i) + 1
    hb = hilb(i)
    free_values = np.dot(hb, original_result)
    print relative_error(original_result, np.linalg.solve(hb, free_values)) / relative_error(original_result, symmetric_system_solve(hb, free_values, (1.0 + sqrt(17))/8))
"""