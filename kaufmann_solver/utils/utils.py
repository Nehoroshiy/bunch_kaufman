import numpy as np
from math import sqrt
from numpy import identity as I
from numpy import tril, triu, dot
from collections import Counter


def transposition_matrix(size, i, j):
    """Creates transposition matrix of given size with given row and column transposition indices.

    Args:
        size (int): size of matrix
        i (int): row index of transposition
        j (int): column index of transposition

    Returns:
        np.array: transposition matrix

    Raises:
        Exception: An error occurred while passing size <= 0
        Exception: An error occurred while passing i >= size
        Exception: An error occurred while passing j >= size
    """
    if size <= 0:
        raise Exception('given non-positive size, expected positive')
    if i >= size:
        raise Exception('given row index (' + str(i) + ') is out of size of matrix (' + str(size) + ')')
    if j >= size:
        raise Exception('given column index (' + str(j) + ') is out of size of matrix (' + str(size) + ')')
    permutation = I(size)
    if i != j:
        permutation[i, j], permutation[i, i] = permutation[i, i], permutation[i, j]
        permutation[j, i], permutation[j, j] = permutation[j, j], permutation[j, i]
    return permutation


def frobenius_norm(matrix):
    """Counts Frobenius norm of a given matrix.

    Args:
         (np.array): matrix for norm counting

    Returns:
        float: Frobenius norm of a matrix

    Raises:
        Exception: An error occurred while passing non-matrix object

    """
    if len(matrix.shape) != 2:
        raise Exception('Passed object is not a 2D matrix and has shape ' + str(matrix.shape))
    sum = 0
    for i in xrange(matrix.shape[0]):
        for j in xrange(matrix.shape[0]):
            sum += matrix[i, j] ** 2
    return sqrt(sum)


def separate_permutation(PL):
    """Separates permutation matrix P and lower triangular matrix L from their multiply PL.

    Args:
        PL (np.array): matrix for separation

    Returns:
        np.matrix: permutation matrix
        np.matrix: lower triangular matrix

    Raises:
        Exception: An error occurred while passing non-PL matrix

    """
    permutation = np.zeros(PL.shape)
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


def triangular_inversion(triang_arg):
    """Counts inversion of a triangular matrix (lower or upper).

        NOT TESTED PROPERLY
        function also trying to predict power of nilpotence (atomic_number)
        of nilpotent matrix inside of an algorithm

    Args:
        triang_arg (np.array): triangular matrix for inversion

    Returns:
        np.matrix: inverse of triangular matrix

    Raises:
        Exception: An error occurred while passing non-square matrix
        Exception: An error occurred while passing non-triangular matrix
        Exception: An error occurred while passing singular matrix

    """
    if len(triang_arg.shape) != 2 or triang_arg.shape[0] != triang_arg.shape[1]:
        raise Exception('Matrix is non-square')
    if not np.array_equal(triang_arg, tril(triang_arg)) and not np.array_equal(triang_arg, triu(triang_arg)):
        raise Exception('Matrix is not triangular')
    if not len(triang_arg.diagonal().nonzero()[0]):
        raise Exception('Matrix is singular')

    triang = triang_arg.copy()
    n = triang.shape[0]

    unitriang_maker = I(n) / triang.diagonal()
    unitriang = dot(unitriang_maker, triang)
    nilpotent = unitriang - I(n)

    z = zip(nilpotent.nonzero()[0], nilpotent.nonzero()[1])
    atomic_number = len(Counter([column for (row, column) in z]))

    unitriang_inverse = I(n)
    for i in xrange(atomic_number):
        unitriang_inverse = I(n) - dot(nilpotent, unitriang_inverse)

    """
    print '-'*80
    print 'check (Tr * Tr^-1):'
    print dot(triang, dot(unitriang_inverse, unitriang_maker))
    print '-'*80
    """

    return dot(unitriang_inverse, unitriang_maker)