import numpy as np
from numpy import float64 as f64
import bigfloat as bf


EXACT_PRECISION = 500


def inverse_1_2_exact(small_matrix, dtype=f64):
    """Performs inverse of small matrices of size 1 or 2 by simple formulas

    Args:
        small_matrix (np.array): matrix for inverse searching

    Returns:
        np.array: inverse of small matrix

    Raises:
        Exception: An error occurred because matrix is not 1x1 or 2x2
        Exception: An error occurred because matrix of size 1x1 or 2x2 is singular
    """
    with bf.Context(precision=EXACT_PRECISION):
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
        det = bf.sub(bf.mul(a, d), bf.mul(b, c))
        #det = a * d - b * c
        if det == 0:
            raise Exception('Matrix of size 2x2 is singular')
        inverse = np.zeros(small_matrix.shape, dtype=dtype)
        inverse[0, 0] = bf.div(d,  det)
        inverse[0, 1] = bf.div(-c,  det)
        inverse[1, 0] = bf.div(-b,  det)
        inverse[1, 1] = bf.div(a,  det)
        return inverse


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


def partial_left_one_exact(matrix, vector, index):
    with bf.Context(precision=EXACT_PRECISION):
        n = matrix.shape[0]
        multiplier = matrix[index]
        for i in xrange(index + 1, n, 1):
            v_i = bf.BigFloat(np.float64(vector[i - index - 1]))
            for j in xrange(n):
                matrix[i, j] = bf.add(np.float64(matrix[i, j]), bf.mul(v_i, np.float64(multiplier[j])))


def partial_left_two(matrix, v1, v2, i1, i2):
    n = matrix.shape[0]
    mul1 = matrix[i1]
    mul2 = matrix[i2]
    low_bound = 2
    for i in xrange(low_bound, n, 1):
        matrix[i] += v1[i - low_bound] * mul1 + v2[i - low_bound] * mul2


def partial_left_two_exact(matrix, v1, v2, i1, i2):
    with bf.Context(precision=EXACT_PRECISION):
        n = matrix.shape[0]
        mul1 = matrix[i1]
        mul2 = matrix[i2]
        low_bound = 2
        for i in xrange(low_bound, n, 1):
            v1_i = bf.BigFloat(v1[i - low_bound])
            v2_i = bf.BigFloat(v2[i - low_bound])
            for j in xrange(n):
                matrix[i, j] = bf.add(matrix[i, j], bf.add(bf.mul(v1_i, mul1[j]), bf.mul(v2_i, mul2[j])))


def partial_right_one(matrix, vector, index):
    n = matrix.shape[0]
    multiplier = matrix[:, index]
    for i in xrange(n):
        matrix[i, index + 1:] += multiplier[i] * vector


def partial_right_one_exact(matrix, vector, index):
    with bf.Context(precision=EXACT_PRECISION):
        n = matrix.shape[0]
        multiplier = matrix[:, index]
        for j in xrange(index + 1, n, 1):
            v_j = bf.BigFloat(np.float64(vector[j - index - 1]))
            for i in xrange(n):
                matrix[i, j] = bf.add(np.float64(matrix[i, j]), bf.mul(v_j, np.float64(multiplier[i])))


def partial_right_two(matrix, v1, v2, i1, i2):
    n = matrix.shape[0]
    mul1 = matrix[:, i1]
    mul2 = matrix[:, i2]
    low_bound = 2
    for i in xrange(n):
        matrix[i, low_bound:] += mul1[i] * v1 + mul2[i] * v2


def partial_right_two_exact(matrix, v1, v2, i1, i2):
    with bf.Context(precision=EXACT_PRECISION):
        n = matrix.shape[0]
        mul1 = matrix[:, i1]
        mul2 = matrix[:, i2]
        low_bound = 2
        for j in xrange(low_bound, n, 1):
            v1_j = bf.BigFloat(v1[j - low_bound])
            v2_j = bf.BigFloat(v2[j - low_bound])
            for i in xrange(n):
                matrix[i, j] = bf.add(matrix[i, j], bf.add(bf.mul(v1_j, mul1[i]), bf.mul(v2_j, mul2[i])))


def tridiagonal_dot_exact(tridiagonal, v, dtype=f64):
    with bf.Context(precision=EXACT_PRECISION):
        n = len(v)
        result = np.zeros(n, dtype=dtype)
        vlist = []
        for i in xrange(n):
            vlist.append(bf.BigFloat(np.float64(v[i])))
        result[0] = bf.add(bf.mul(np.float64(tridiagonal[1, 0]), vlist[0]), bf.mul(np.float64(tridiagonal[0, 1]), vlist[1]))
        for i in xrange(1, n - 1, 1):
            result[i] = bf.add(bf.add(bf.mul(np.float64(tridiagonal[2, i - 1]), vlist[i - 1]), bf.mul(np.float64(tridiagonal[1, i]), vlist[i])), bf.mul(np.float64(tridiagonal[0, i + 1]), vlist[i + 1]))
        result[n - 1] = bf.add(bf.mul(np.float64(tridiagonal[2, n - 2]), vlist[n - 2]), bf.mul(np.float64(tridiagonal[1, n - 1]), vlist[n - 1]))
        return result

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


def tridiagonal_inversion_exact(tridiagonal, cell_sizes, dtype=np.float64):
    with bf.Context(precision=EXACT_PRECISION):
        sum = 0
        inverse = np.zeros(tridiagonal.shape, dtype=dtype)
        for cs in cell_sizes:
            if cs == 1:
                inverse[1, sum] = bf.div(1, np.float64(tridiagonal[1, sum]))
            else:
                a = np.float64(tridiagonal[1, sum])
                b = np.float64(tridiagonal[0, sum + 1])
                c = np.float64(tridiagonal[1, sum])
                d = np.float64(tridiagonal[1, sum + 1])
                #det = a * d - b * c
                det = bf.sub(bf.mul(a, d), bf.mul(b, c))
                inverse[1, sum] = bf.div(d, det)
                inverse[0, sum + 1] = bf.div(-c, det)
                inverse[1, sum] = bf.div(-b, det)
                inverse[1, sum + 1] = bf.div(a, det)
            sum += cs
        return inverse


def permutation_and_lower():
    pass


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
    L = permutation.T.dot(PL)
    #diag = L.diagonal()
    #for i in xrange(PL.shape[0]):
    #    L[i] /= diag[i]
    #    print 'L modification:'
    #    print L

    return permutation, L