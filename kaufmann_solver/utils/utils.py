import numpy as np
from math import sqrt
from numpy import identity as I
from numpy import tril, triu, dot, float128 as f128
from operator import itemgetter


def euclid_vector_norm(vector):
    """Counts Euclid norm of a given vector.

    Args:
         (np.array): vector for norm counting

    Returns:
        float: Euclid norm of a vector

    Raises:
        Exception: An error occurred while passing non-vector object

    """
    if len(vector.shape) != 1:
        raise Exception('vector must have 1 dimension!')
    s = sum([x**2 for x in vector])
    return sqrt(s)


def relative_error(original_vector, computed_vector):
    """Counts Euclid norm of a given vector.

    Args:
         (np.array): vector for norm counting

    Returns:
        float: Euclid norm of a vector

    Raises:
        Exception: An error occurred while passing non-vector object to first or second arg
        Exception: An error occurred while passing vectors with different shapes
        Exception: An error occurred while passing zero-norm original vector
    """
    if len(original_vector.shape) != 1 or len(computed_vector.shape) != 1:
        raise Exception('vectors must have 1 dimension!')
    if original_vector.shape != computed_vector.shape:
        raise Exception('vectors must have equal shapes!')
    if euclid_vector_norm(original_vector) == 0:
        raise Exception('Original vector must have non-zero norm')
    return euclid_vector_norm(original_vector - computed_vector) / euclid_vector_norm(original_vector)


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


def max_pseudo_norm(matrix):
    return abs(matrix).max()


def triangular_inversion(triang_arg):
    """Counts inversion of a triangular matrix (lower or upper).

        NOT TESTED PROPERLY
        function trying to predict form of inversion and count it without heavy computations
        SEE gauss atomic triangular matrix and it's block analog

        NOT TESTED PROPERLY
        function also trying to predict power of nilpotence (atomic_number)
        of nilpotent matrix inside of an algorithm

    Args:
        triang_arg (np.array): triangular matrix for inversion

    Returns:
        np.array: inverse of triangular matrix

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

    # possibility of simple inversion prediction
    z = zip(nilpotent.nonzero()[0], nilpotent.nonzero()[1])
    if z[0][0] > z[0][1]:
        # lower triangular case
        i_min = min(z, key=itemgetter(0))[0]
        j_max = max(z, key=itemgetter(1))[1]
        if i_min > j_max:
            return dot(I(n) - nilpotent, unitriang_maker)
        else:
            atomic_number = len(set([column for (row, column) in z]))
            #atomic_number = len(Counter([column for (row, column) in z]))
    else:
        # upper triangular case
        i_max = max(z, key=itemgetter(0))[0]
        j_min = min(z, key=itemgetter(1))[1]
        if j_min > i_max:
            return dot(I(n) - nilpotent, unitriang_maker)
        else:
            atomic_number = len(set([row for (row, column) in z]))

    # nilpotence power prediction
    # not sure that it will work for U instead of L
    #atomic_number = len(Counter([column for (row, column) in z]))

    unitriang_inverse = I(n)
    for i in xrange(atomic_number):
        unitriang_inverse = I(n) - dot(nilpotent, unitriang_inverse)
    return dot(unitriang_inverse, unitriang_maker)