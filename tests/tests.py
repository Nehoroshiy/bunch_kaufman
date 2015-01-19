import numpy as np
from numpy import dot
from math import factorial
from kaufmann_solver.utils.utils import frobenius_norm, euclid_vector_norm
from kaufmann_solver.bunch_kaufmann import bunch_kaufmann, symmetric_system_solve

# hilbert matrix (Hij = 1/(i + j - 1))

def boundline(n=80):
    print '-'*n

def binomial(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return factorial(n) // (factorial(k) * factorial(n-k))

def hilb(n, m=0):
    if n < 1 or m < 0:
        raise ValueError("Matrix size must be one or greater")
    elif n == 1 and (m == 0 or m == 1):
        return np.array([[1]])
    elif m == 0:
        m = n
    return 1. / (np.arange(1, n + 1) + np.arange(0, m)[:, np.newaxis])


def invhilb(n):
    H = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = ((-1)**(i + j)) * (i + j + 1) * binomial(n + i, n - j - 1) * \
            binomial(n + j, n - i - 1) * binomial(i + j, i) ** 2
    return H


def factorization_test(mtx):
    """Tests Bunch-Kaufman factorization for a given matrix.

        Test by factorizing it, restoring from factors and counting difference.

    Args:
         (np.array): testing matrix

    Returns:
        none

    Raises:
        Exception: An error occurred when Bunch-Kaufman doesn't work properly.

    """
    tridiagonal, P, L, cell_sizes = bunch_kaufmann(mtx.copy())
    if filter(lambda x: x != 1 and x != 2, cell_sizes):
        raise Exception('Cell sizes in Bunch-Kaufman must be 1-2')
    if not np.array_equal(L, np.tril(L)):
        raise Exception('Bunch-Kaufman algo must make lower triangular matrix')
    assembled_result = dot(dot(dot(dot(P, L), tridiagonal), np.matrix(L).getH()), P.T)
    boundline()
    print 'This is Bunch-Kaufman test.'
    print 'Frobenius norm of difference:'
    print frobenius_norm(mtx - assembled_result)
    boundline()


def linear_solve_test(mtx, original_solve=[]):
    """Tests Bunch-Kaufman-based symmetric system solver for a given matrix.

    Args:
         (np.array): testing matrix
         (np.array): original solve for testing

    Returns:
        none

    Raises:
        Exception: An error occurred when sizes of matrix and original solution doesn't fit.

    """
    if not original_solve:
        original_solve = np.zeros(mtx.shape[0]) + 1
    if original_solve.shape[0] != mtx.shape[0]:
        raise Exception('Sizes of matrix and original solve must be equal!')
    free_variables = mtx.dot(original_solve)

    kaufmann_result = symmetric_system_solve(mtx, free_variables)

    boundline()
    print 'This is linear symmetric system solver test.'
    print 'Original result:'
    print original_solve
    print 'Kaufmann result:'
    print kaufmann_result

    print 'Euclid norm of delta:'
    print euclid_vector_norm(kaufmann_result - original_solve)
    boundline()