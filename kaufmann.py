import numpy as np
np.set_printoptions(precision=15, suppress=True)

from kaufmann_solver.utils.utils import frobenius_norm, relative_error
from tests.algorithm_tuner import vizualize_bunch_kaufman_tune, get_optimal_alpha
from tests.tests import hilb
from tests.algorithm_tuner import hilbert_matrix_test
from kaufmann_solver.bunch_kaufmann import bunch_kaufmann, symmetric_system_solve, symmetric_system_solve_without_refinement
from math import sqrt



mtx = hilb(13)
#mtx = np.loadtxt('matrix.txt')
#mtx = np.loadtxt('test_tridiagonal.txt')

etalon = mtx.copy()

open("result.txt", "w").write(str(mtx))

stripped_matrix, P, L, cell_sizes = bunch_kaufmann(mtx, (1.0 + sqrt(17.0))/8)

mtx = etalon.copy()
free_variables = mtx.dot(np.zeros(mtx.shape[0]) + 1)

kaufmann_result = symmetric_system_solve_without_refinement(mtx, free_variables)
print '-'*80
print 'Kaufmann result:'
print kaufmann_result

mtx = etalon.copy()
