import numpy as np
from math import sqrt
from numpy import float128


def transposition_matrix(size, idx1, idx2):
    permutation = np.identity(size)
    if idx1 != idx2:
        permutation[:, [idx1, idx2]] = permutation[:, [idx2, idx1]]
    return permutation


def frobenius_norm(mtx):
    #print mtx.shape
    sum = 0
    for i in xrange(mtx.shape[0]):
        for j in xrange(mtx.shape[0]):
            sum += mtx[i, j] ** 2
    return sqrt(sum)
    #return sqrt(sum([mtx[i, j]**2 for i in mtx.shape[0] for j in mtx.shape[1]]))


def separate_permutation(PL):
    permutation = np.zeros(PL.shape)

    z = zip(PL.nonzero()[0], PL.nonzero()[1])
    by_index = [[] for _ in xrange(PL.shape[0])]
    for (i, j) in z:
        by_index[i].append((i, j))

    for t in xrange(PL.shape[0]):
        by_index = map(lambda x: x if len(x) == 1 else filter(lambda (i, j): j != t, x), by_index)

    assert all(len(x) == 1 for x in by_index), \
        "Something wrong with PL matrix. (I can't simply guess the P-matrix and L-matrix)"

    for [(i, j)] in by_index:
        permutation[i][j] = 1

    return permutation, permutation.T.dot(PL)