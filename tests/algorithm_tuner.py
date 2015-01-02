from kaufmann_solver.utils.utils import relative_error
from kaufmann_solver.bunch_kaufmann import symmetric_system_solve
from tests import hilb
from numpy import dot, zeros
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from math import log
from operator import itemgetter


def hilbert_matrix_test(max_size):
    relative_errors = zeros(max_size)
    for i in xrange(1, max_size, 1):
        hb = hilb(i)
        original_result = zeros(i) + 1.
        free_values = dot(hb, original_result)
        computed_result = symmetric_system_solve(hb, free_values)
        relative_errors[i] = relative_error(original_result, computed_result)

    plt.plot(xrange(max_size), relative_errors, alpha=0.3)
    plt.show()


def bunch_kaufman_tune(matrix_orig, original_result, lattice_size):
    free_values = dot(matrix_orig, original_result)
    alphas = zeros(lattice_size - 1)
    relative_errors = zeros(lattice_size - 1)
    for i in xrange(0, lattice_size - 1, 1):
        alphas[i] = 1.0*(i + 1)/lattice_size
        computed_result = symmetric_system_solve(matrix_orig, free_values, alphas[i])
        relative_errors[i] = relative_error(original_result, computed_result)
    return alphas, relative_errors

def get_optimal_alpha(matrix_orig, original_result, left, right, delta, eps=1e-6):
    print left, right
    if right - left < eps:
        return (right + left) / 2
    free_values = dot(matrix_orig, original_result)
    lattice_size = 200
    n = len(xrange(1, lattice_size, 1))
    length = right - left
    alphas = zeros(n)
    relative_errors = zeros(n)
    for i in xrange(lattice_size - 1):
        alphas[i] = left + length * (1.0 * (i + 1) / lattice_size)
        computed_result = symmetric_system_solve(matrix_orig, free_values, alphas[i])
        relative_errors[i] = relative_error(original_result, computed_result)
    [min_val, idx] = min([(relative_errors[i], i) for i in xrange(lattice_size - 1)], key=itemgetter(0))
    new_left = max(left + length * (1.0 * idx / lattice_size) * (1.0 - delta), left)
    new_right = min(left + length * (1.0 * idx / lattice_size) * (1.0 + delta), right)
    return get_optimal_alpha(matrix_orig, original_result, new_left, new_right, eps)

def vizualize_bunch_kaufman_tune(matrix_orig, n):
    original_result = zeros(matrix_orig.shape[0]) + 1
    lattice_size = 4
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in xrange(n):
        lattice_size *= 2
        lattices = zeros(lattice_size - 1) + log(lattice_size, 2)
        alphas, relative_errors = bunch_kaufman_tune(matrix_orig, original_result, lattice_size)
        #print relative_errors
        ax.plot(alphas, lattices, relative_errors, alpha=0.3)

    plt.show()

