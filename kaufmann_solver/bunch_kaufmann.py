import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=160)
from numpy import identity as I, dot, float128 as f128
from math import sqrt
from kaufmann_solver.utils.bunch_kaufman_utils import *


def bunch_kaufman(mtx_origin, alpha=(1. + sqrt(17)) / 8, regularize=False, regularize_coeff=1e-4):
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
    if alpha <= 0 or alpha >= 1:
        raise Exception("alpha must be in range (0, 1), but alpha = " + str(alpha))
    if len(mtx_origin.shape) != 2:
        raise Exception("Matrix isn't 2D")
    if mtx_origin.shape[0] != mtx_origin.shape[1]:
        raise Exception('Square matrix expected, matrix of shape ' + str(mtx_origin.shape) + ' is given')
    if not np.array_equal(np.array(np.matrix(mtx_origin).getH()), mtx_origin):
        raise Exception('Self-conjugate matrix expected')

    mtx = np.array(mtx_origin, dtype=f128)
    n = mtx.shape[0]
    tridiagonal = np.zeros([3, n], dtype=f128)
    sum = 0
    cell_sizes = []
    PL = I(n, dtype=f128)
    flip = False
    while sum < n:
        m = n - sum
        mtx_view = mtx[sum: n, sum: n]
        if flip:
            mtx_view[:, :] = mtx_view[::-1, ::-1]
            """print '-'*80
            print 'PL before flipping'
            print PL
            PL_C = PL.copy()
            print '-'*80"""
            PL[:, sum:] = PL[:, :sum - 1:-1]
            """print 'PL after flipping'
            print PL
            print '-'*80"""
        idx = np.argmax(np.abs(mtx_view.diagonal()))
        swap_indices = (0, idx)
        triangular = I(n, dtype=f128)
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
        T_k_inverse = inverse_1_2(T_k, dtype=f128)
        B_k = mtx_view[n_k: m, 0: n_k]
        triangular_view[n_k: m, 0: n_k] = dot(-B_k, T_k_inverse)

        PL_view = PL[sum:n, sum:n]
        if n_k == 1:
            tridiagonal[1, sum] = T_k[0, 0]
            tri_one = triangular_view[1: m, 0]
            """print '-'*80
            print 'mtx before left mul'
            print mtx_view"""
            partial_left_one(mtx_view, tri_one, 0)
            """print '-'*80
            print 'mtx before right mul, after left'
            print mtx_view"""
            partial_right_one(mtx_view, tri_one, 0)
            """print '-'*80
            print 'mtx after right mul'
            print mtx_view
            print '-'*80"""
            mtx_view[1: m, 0] = 0
            mtx_view[0, 1: m] = 0
            """print '-'*80
            print 'PL before right mul'
            print PL
            print '-'*80"""
            for i in xrange(n):
                PL[i, sum] += dot(PL[i, sum + 1:n], (-tri_one))
            """print '-'*80
            print 'PL after right mul'
            print PL
            print '-'*80"""
        else:
            tridiagonal[1, sum], tridiagonal[1, sum + 1] = T_k[0, 0], T_k[1, 1]
            tridiagonal[0, sum + 1] = T_k[0, 1]
            tridiagonal[2, sum] = T_k[1, 0]
            tri_one = triangular_view[2: m, 0]
            tri_two = triangular_view[2: m, 1]
            """print '-'*80
            print 'mtx before left mul'
            print mtx_view"""
            partial_left_two(mtx_view, tri_one, tri_two, 0, 1)
            """print '-'*80
            print 'mtx before right mul, after left'
            print mtx_view"""
            partial_right_two(mtx_view, tri_one, tri_two, 0, 1)
            """print '-'*80
            print 'mtx after right mul'
            print mtx_view
            print '-'*80"""
            mtx_view[2: m, [0, 1]] = 0
            mtx_view[[0, 1], 2: m] = 0
            """print '-'*80
            print 'PL before right mul'
            print PL
            print '-'*80"""
            for i in xrange(n):
                PL[i, sum] += dot(PL[i, sum + 2:n], (-tri_one))
                PL[i, sum + 1] += dot(PL[i, sum + 2:n], (-tri_two))
            """print '-'*80
            print 'PL after right mul'
            print PL
            print '-'*80"""
        sum += n_k
        cell_sizes.append(n_k)
        flip = not flip
    #print 'ASSEMBLED:'
    #print dot(dot(PL, mtx), np.matrix(PL).getH())
    P, L = separate_permutation(PL, dtype=f128)
    if regularize:
        for w, (i, j) in enumerate(zip(*tridiagonal.nonzero())):
            if tridiagonal[i, j] < regularize_coeff:
                tridiagonal[i, j] = regularize_coeff if w % 2 else regularize_coeff * 1.1
        for i in xrange(n-1):
            tridiagonal[2, i] = tridiagonal[0, i + 1]
        for i in xrange(n):
            if tridiagonal[1, i] == 0:
                tridiagonal[1, i] = regularize_coeff
        #for i in xrange(n):
        #    tridiagonal[1, i] *= 1.5

    return P, L, cell_sizes, tridiagonal


def bunch_kaufman_flipper(mtx_origin, alpha=(1. + sqrt(17)) / 8, regularize=False, regularize_coeff=1e-4):
    mtx = np.array(mtx_origin, dtype=f128)
    n = mtx.shape[0]
    tridiagonal = np.zeros([3, n], dtype=f128)
    sum_left = 0
    sum_right = 0
    cell_sizes_left = []
    cell_sizes_right = []
    PL = I(n, dtype=f128)
    flip = False
    while sum_left < n - sum_right:
        m = n - sum_left - sum_right
        mtx_view = mtx[sum_left: n - sum_right, sum_left: n - sum_right]
        if flip:
            f_idx = m - 1
            s_idx = m - 2
        else:
            f_idx = 0
            s_idx = 1
        idx = np.argmax(np.abs(mtx_view.diagonal()))
        swap_indices = (f_idx, idx)
        triangular = I(n, dtype=f128)
        triangular_view = triangular[sum_left: n - sum_right, sum_left: n - sum_right]
        exchange_rows(mtx_view, swap_indices[0], swap_indices[1])
        exchange_columns(mtx_view, swap_indices[0], swap_indices[1])
        exchange_columns(PL, swap_indices[0] + sum_left, swap_indices[1] + sum_left)

        idx = np.argmax(np.abs(mtx_view[:, f_idx]))
        lambda_val = abs(mtx_view[:, f_idx][idx])

        if abs(mtx_view[f_idx, f_idx]) >= alpha * lambda_val:
            n_k = 1
            swap_indices = (f_idx, f_idx)
        else:
            testing_column = np.abs(mtx_view[:, idx])
            testing_column[idx] = 0
            j_idx = np.argmax(testing_column)
            sigma_val = testing_column[j_idx]

            if sigma_val * abs(mtx_view[f_idx, f_idx]) >= alpha * lambda_val**2:
                n_k = 1
                swap_indices = (f_idx, f_idx)
            else:
                if abs(mtx_view[idx][idx]) >= alpha * sigma_val:
                    n_k = 1
                    swap_indices = (f_idx, idx)
                else:
                    n_k = 2
                    swap_indices = (s_idx, idx)
        if n_k == 2:
            exchange_rows(mtx_view, f_idx, j_idx)
            exchange_columns(mtx_view, f_idx, j_idx)
            exchange_columns(PL, sum_left + f_idx, sum_left + j_idx)
        exchange_rows(mtx_view, swap_indices[0], swap_indices[1])
        exchange_columns(mtx_view, swap_indices[0], swap_indices[1])
        exchange_columns(PL, sum_left + swap_indices[0], sum_left + swap_indices[1])

        T_k = mtx_view[0:n_k, 0:n_k]
        if n - sum_right <= sum_left + n_k:
            if n_k == 1:
                tridiagonal[1, sum_left + f_idx] = T_k[0, 0]
            else:
                tridiagonal[1, sum_left + f_idx], tridiagonal[1, sum_left + s_idx] = T_k[1, 1], T_k[0, 0]
                tridiagonal[0, sum_left + f_idx] = T_k[0, 1]
                tridiagonal[2, sum_left + s_idx] = T_k[1, 0]
            cell_sizes_left.append(n_k)
            cell_sizes_right.reverse()
            cell_sizes = cell_sizes_left + cell_sizes_right
            break
        T_k_inverse = inverse_1_2(T_k, dtype=f128)
        if flip:
            B_k = mtx_view[0: m - n_k, m - n_k: m]
            triangular_view[0: m - n_k, m - n_k: m] = dot(-B_k, T_k_inverse)
            sum_right += n_k
            cell_sizes_right.append(n_k)
        else:
            B_k = mtx_view[n_k: m, 0: n_k]
            triangular_view[n_k: m, 0: n_k] = dot(-B_k, T_k_inverse)
            sum_left += n_k
            cell_sizes_left.append(n_k)
        mtx_view[:, :] = dot(dot(triangular_view, mtx_view), np.matrix(triangular_view).getH())[:,:]
        PL = dot(PL, -triangular + 2*I(n))
        flip = not flip
    P, L = separate_permutation(PL, dtype=f128)
    return P, L, cell_sizes, tridiagonal
