import numpy as np


def transposition_matrix(size, idx1, idx2):
    permutation = np.identity(size)
    if idx1 != idx2:
        permutation[idx1][idx2] = 1
        permutation[idx2][idx1] = 1
        permutation[idx1][idx1] = 0
        permutation[idx2][idx2] = 0
    return permutation