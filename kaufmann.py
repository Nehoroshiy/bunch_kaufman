import numpy as np
np.set_printoptions(precision=5, suppress=True)

from kaufmann_solver.utils.tests import hilb
from kaufmann_solver.utils.utils import frobenius_norm
from kaufmann_solver.bunch_kaufmann import bunch_kaufmann, linear_system_solve
from math import sqrt



mtx = hilb(5)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')

etalon = mtx.copy()

open("result.txt", "w").write(str(mtx))

stripped_matrix, PL, cell_sizes = bunch_kaufmann(mtx, (1.0 + sqrt(17.0))/8)

print 'Matrix:'
print etalon
print '-'*80
print 'Check:'
result = PL.dot(stripped_matrix).dot(np.matrix(PL).getH())
print result
print cell_sizes
print '-'*80

print '-'*80
print 'Resulted matrix'
print stripped_matrix
print stripped_matrix.nonzero()
print '-'*80
print 'And PL matrix:'
print PL


print etalon.shape
print result.shape

print frobenius_norm(etalon - result)


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

kaufmann_result = linear_system_solve(mtx, free_variables, 0.9)
print 'Kaufmann linear solver:'
print mtx.dot(kaufmann_result)
print kaufmann_result, free_variables
print 'frobenius norm of difference:'
print kaufmann_result.shape, free_variables.shape
print sqrt(sum([(first - second)**2 for (first, second) in zip(kaufmann_result, free_variables)]))
print '-'*80
print '\n'

ordinary_result = np.linalg.solve(mtx, free_variables)
print 'Ordinary linear solver:'
print mtx.dot(ordinary_result)
print ordinary_result, free_variables
print 'frobenius norm of difference:'
print sqrt(sum([(first - second)**2 for (first, second) in zip(ordinary_result, free_variables)]))
#print frobenius_norm(kaufmann_result - ordinary_result)

#mtx = hilb(8)
#print '\n'*5
#print '-'*80
#print 'matrix:'
#print mtx
#print '-'*80
#print 'linear system solver:'
#print
