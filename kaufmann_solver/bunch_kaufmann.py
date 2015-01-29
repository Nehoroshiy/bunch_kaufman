import numpy as np
np.set_printoptions(precision=15, suppress=True, linewidth=160)
from numpy import identity as I, dot
from math import sqrt
from kaufmann_solver.utils.bunch_kaufman_utils import *


def bunch_kaufman(mtx, alpha=(1. + sqrt(17)) / 8):
    """Performs Bunch-Kaufman factorization of self-conjugate matrix A:
    A = P L T L^* P^t,
    P - permutation matrix,
    L - lower triangular matrix,
    T - tridiagonal matrix

    Args:
        mtx_origin (np.array): matrix for factorization
        alpha (float): tuning coefficient for Bunch-Kaufmann algorithm, 0 < alpha < 1
        in practice best value of alpha is (1. + sqrt(17))/8

    Returns:
        np.array: tridiagonal symmetric matrix T
        np.array: permutation matrix P
        np.array: lower triangular matrix L
        list: list of block sizes of cells in T

    Raises:
        Exception: An error occurred because alpha goes beyond the range (0, 1)
        Exception: An error occurred while passing non-2D-matrix argument
        Exception: An error occurred while passing non-square matrix
        Exception: An error occurred while passing non-self-conjugate matrix
    """
    """if alpha <= 0 or alpha >= 1:
        raise Exception("alpha must be in range (0, 1), but alpha = " + str(alpha))
    if len(mtx_origin.shape) != 2:
        raise Exception("Matrix isn't 2D")
    if mtx_origin.shape[0] != mtx_origin.shape[1]:
        raise Exception('Square matrix expected, matrix of shape ' + str(mtx_origin.shape) + ' is given')
    if not np.array_equal(np.array(np.matrix(mtx_origin).getH()), mtx_origin):
        raise Exception('Self-conjugate matrix expected')
    dtype = mtx_origin.dtype"""
    dtype=mtx.dtype

    #mtx = np.array(mtx_origin, dtype=dtype)
    n = mtx.shape[0]
    tridiagonal = np.zeros([3, n], dtype=dtype)
    sum = 0
    cell_sizes = []
    PL = I(n, dtype=dtype)
    flip = False
    while sum < n:
        m = n - sum
        mtx_view = mtx[sum: n, sum: n]
        if sum >= n - 2:
            if sum == n - 2:
                cell_sizes.append(2)
                tridiagonal[1, sum], tridiagonal[1, sum + 1] = mtx_view[0, 0], mtx_view[1, 1]
                tridiagonal[0, sum + 1] = mtx_view[0, 1]
                tridiagonal[2, sum] = mtx_view[1, 0]
            else:
                tridiagonal[1, sum] = mtx_view[0, 0]
            break
        if flip:
            mtx_view[:, :] = mtx_view[::-1, ::-1]
            PL[:, sum:] = PL[:, :sum - 1:-1]
        idx = np.argmax(np.abs(mtx_view.diagonal()))
        swap_indices = (0, idx)
        triangular = I(n, dtype=dtype)
        triangular_view = triangular[sum: n, sum: n]

        exchange_rows(mtx_view, swap_indices[0], swap_indices[1])
        exchange_columns(mtx_view, swap_indices[0], swap_indices[1])
        exchange_columns(PL, swap_indices[0] + sum, swap_indices[1] + sum)

        idx = np.argmax(np.abs(mtx_view[:, 0]))
        lambda_val = abs(mtx_view[:, 0][idx])
        if abs(mtx_view[0, 0]) >= alpha * lambda_val:
            n_k = 1
            swap_indices = (0, 0)
        else:
            testing_column = np.abs(mtx_view[:, idx])
            testing_column[idx] = 0
            j_idx = np.argmax(testing_column)
            sigma_val = testing_column[j_idx]

            if sigma_val * abs(mtx_view[0][0]) >= alpha * lambda_val**2:
                n_k = 1
                swap_indices = (0, 0)
            else:
                if abs(mtx_view[idx][idx]) >= alpha * sigma_val:
                    n_k = 1
                    swap_indices = (0, idx)
                else:
                    n_k = 2
                    assert idx != j_idx, 'please check your factorization. This indices MUST be different'
                    swap_indices = (1, idx)
        if n_k == 2:
            exchange_rows(mtx_view, 0, j_idx)
            exchange_columns(mtx_view, 0, j_idx)
            exchange_columns(PL, sum + 0, sum + j_idx)
        exchange_rows(mtx_view, swap_indices[0], swap_indices[1])
        exchange_columns(mtx_view, swap_indices[0], swap_indices[1])
        exchange_columns(PL, sum + swap_indices[0], sum + swap_indices[1])

        T_k = mtx_view[0:n_k, 0:n_k]
        if n <= sum + n_k:
            if n_k == 1:
                tridiagonal[1, sum] = T_k[0, 0]
            else:
                tridiagonal[1, sum], tridiagonal[1, sum + 1] = T_k[0, 0], T_k[1, 1]
                tridiagonal[0, sum + 1] = T_k[0, 1]
                tridiagonal[2, sum] = T_k[1, 0]
            cell_sizes.append(n_k)
            break
        T_k_inverse = inverse_1_2(T_k, dtype=dtype)
        B_k = mtx_view[n_k: m, 0: n_k]
        triangular_view[n_k: m, 0: n_k] = dot(-B_k, T_k_inverse)

        PL_view = PL[sum:n, sum:n]
        if n_k == 1:
            tridiagonal[1, sum] = T_k[0, 0]
            tri_one = triangular_view[1: m, 0]
            partial_left_one(mtx_view, tri_one, 0)
            partial_right_one(mtx_view, tri_one, 0)
            mtx_view[1: m, 0] = 0
            mtx_view[0, 1: m] = 0
            for i in xrange(n):
                PL[i, sum] += dot(PL[i, sum + 1:n], (-tri_one))
        else:
            tridiagonal[1, sum], tridiagonal[1, sum + 1] = T_k[0, 0], T_k[1, 1]
            tridiagonal[0, sum + 1] = T_k[0, 1]
            tridiagonal[2, sum] = T_k[1, 0]
            tri_one = triangular_view[2: m, 0]
            tri_two = triangular_view[2: m, 1]
            partial_left_two(mtx_view, tri_one, tri_two, 0, 1)
            partial_right_two(mtx_view, tri_one, tri_two, 0, 1)
            mtx_view[2: m, [0, 1]] = 0
            mtx_view[[0, 1], 2: m] = 0
            for i in xrange(n):
                PL[i, sum] += dot(PL[i, sum + 2:n], (-tri_one))
                PL[i, sum + 1] += dot(PL[i, sum + 2:n], (-tri_two))
        sum += n_k
        cell_sizes.append(n_k)
    P, L = separate_permutation_fast(PL, dtype=dtype)

    return P, L, cell_sizes, tridiagonal
