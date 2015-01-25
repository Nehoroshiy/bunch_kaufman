import numpy as np
from numpy import float64 as f64


def inverse_1_2(small_matrix, dtype=f64):
    """Performs inverse of small matrices of size 1 or 2 by simple formulas

    Args:
        small_matrix (np.array): matrix for inverse searching

    Returns:
        np.array: inverse of small matrix

    Raises:
        Exception: An error occurred because matrix is not 1x1 or 2x2
        Exception: An error occurred because matrix of size 1x1 or 2x2 is singular
    """
    if small_matrix.shape == (1, 1):
        if small_matrix[0, 0] == 0:
            raise Exception('Matrix of size 1x1 is singular')
        return np.array([[1.0 / small_matrix[0, 0]]], dtype=dtype)
    if small_matrix.shape != (2, 2):
        raise Exception("Matrix isn't 2x2 matrix")
    a = small_matrix[0, 0]
    b = small_matrix[0, 1]
    c = small_matrix[1, 0]
    d = small_matrix[1, 1]
    det = a * d - b * c
    if det == 0:
        raise Exception('Matrix of size 2x2 is singular')
    inverse = np.zeros(small_matrix.shape, dtype=dtype)
    inverse[0, 0] = d / det
    inverse[0, 1] = -c / det
    inverse[1, 0] = -b / det
    inverse[1, 1] = a / det
    return inverse


def identity_permutation(n):
    return np.array([i for i in xrange(n)])


def transpose_permutation(p):
    transpose = np.zeros(len(p))
    for (i, p_i) in enumerate(p):
        transpose[p_i] = i
    return transpose


def compose_permutations(outer, inner):
    compose = np.zeros(len(outer))
    for i in xrange(len(outer)):
        compose[i] = outer[inner[i]]
    return compose


def exchange_rows(matrix, idx1, idx2):
    if idx1 == idx2:
        return
    matrix[[idx1, idx2]] = matrix[[idx2, idx1]]


def exchange_columns(matrix, idx1, idx2):
    if idx1 == idx2:
        return
    matrix[:, [idx1, idx2]] = matrix[:, [idx2, idx1]]


def partial_left_one(matrix, vector, index):
    n = matrix.shape[0]
    multiplier = matrix[index]
    for i in xrange(index + 1, n, 1):
        matrix[i] += vector[i - index - 1] * multiplier


def partial_left_two(matrix, v1, v2, i1, i2):
    n = matrix.shape[0]
    mul1 = matrix[i1]
    mul2 = matrix[i2]
    low_bound = 2
    for i in xrange(low_bound, n, 1):
        matrix[i] += v1[i - low_bound] * mul1 + v2[i - low_bound] * mul2


def partial_right_one(matrix, vector, index):
    n = matrix.shape[0]
    multiplier = matrix[:, index]
    for i in xrange(n):
        matrix[i, index + 1:] += multiplier[i] * vector


def partial_right_two(matrix, v1, v2, i1, i2):
    n = matrix.shape[0]
    mul1 = matrix[:, i1]
    mul2 = matrix[:, i2]
    low_bound = 2
    for i in xrange(n):
        matrix[i, low_bound:] += mul1[i] * v1 + mul2[i] * v2


def tridiagonal_dot(tridiagonal, v, dtype=f64):
    n = len(v)
    result = np.zeros(n, dtype=dtype)
    result[0] = tridiagonal[1, 0] * v[0] + tridiagonal[0, 1] * v[1]
    for i in xrange(1, n - 1, 1):
        result[i] = tridiagonal[2, i - 1] * v[i - 1] + tridiagonal[1, i] * v[i] + tridiagonal[0, i + 1] * v[i + 1]
    result[n - 1] = tridiagonal[2, n - 2] * v[n - 2] + tridiagonal[1, n - 1] * v[n - 1]
    return result


def tridiagonal_inversion(tridiagonal, cell_sizes, dtype=np.float64):
    sum = 0
    inverse = np.zeros(tridiagonal.shape, dtype=dtype)
    for cs in cell_sizes:
        if cs == 1:
            inverse[1, sum] = 1. / tridiagonal[1, sum]
        else:
            a = tridiagonal[1, sum]
            b = tridiagonal[0, sum + 1]
            c = tridiagonal[1, sum]
            d = tridiagonal[1, sum + 1]
            det = a * d - b * c
            inverse[1, sum] = d / det
            inverse[0, sum + 1] = -c / det
            inverse[1, sum] = -b / det
            inverse[1, sum + 1] = a / det
        sum += cs
    return inverse


def separate_permutation(PL, dtype=np.float64):
    """Separates permutation matrix P and lower triangular matrix L from their multiply PL.

    Args:
        PL (np.array): matrix for separation

    Returns:
        np.matrix: permutation matrix
        np.matrix: lower triangular matrix

    Raises:
        Exception: An error occurred while passing non-PL matrix

    """
    permutation = np.zeros(PL.shape, dtype=dtype)
    z = zip(PL.nonzero()[0], PL.nonzero()[1])
    by_index = [[] for _ in xrange(PL.shape[0])]
    for (i, j) in z:
        by_index[i].append((i, j))

    for t in xrange(PL.shape[0]):
        by_index = map(lambda x: x if len(x) == 1 else filter(lambda (i, j): j != t, x), by_index)

    if not all(len(x) == 1 for x in by_index):
        raise Exception("Matrix isn't PL")

    for [(i, j)] in by_index:
        permutation[i][j] = 1

    return permutation, permutation.T.dot(PL)