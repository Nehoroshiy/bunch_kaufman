import numpy as np
np.set_printoptions(precision=4, suppress=True)

from kaufmann_solver.utils.tests import hilb
from kaufmann_solver.bunch_kaufmann import bunch_kaufmann



mtx = hilb(8)

open("result.txt", "w").write(str(mtx))


print mtx

stripped_matrix, P, cell_sizes = bunch_kaufmann(mtx, 0.5)

print '-'*80
print 'Check:'
print np.matrix(P).getI().dot(stripped_matrix).dot(np.matrix(P).getH().getI())
print '-'*80

print '-'*80
print 'Resulted matrix'
print stripped_matrix
