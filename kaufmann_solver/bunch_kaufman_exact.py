import numpy as np
np.set_printoptions(precision=15, suppress=True, linewidth=160)
from numpy import identity as I, dot, float128 as f128
from math import sqrt
from kaufmann_solver.utils.bunch_kaufman_utils import *
from kaufmann_solver.utils.utils import euclid_vector_norm, frobenius_norm, max_pseudo_norm
from scipy import sparse
import bigfloat as bf


def bunch_kaufman_exact(mtx_origin, alpha=(1. + sqrt(17)) / 8, regularize=False, regularize_coeff=1e-4):
    dtype = mtx_origin.dtype
    mtx = np.array(mtx_origin, dtype=dtype)
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
        T_k_inverse = inverse_1_2_exact(T_k, dtype=dtype)
        B_k = mtx_view[n_k: m, 0: n_k]
        triangular_view[n_k: m, 0: n_k] = dot(-B_k, T_k_inverse)

        PL_view = PL[sum:n, sum:n]
        if n_k == 1:
            tridiagonal[1, sum] = T_k[0, 0]
            tri_one = triangular_view[1: m, 0]
            tri_one = -B_k[:, 0] * T_k_inverse[0, 0]
            """print '-'*80
            print 'mtx before left mul'
            print mtx_view"""
            partial_left_one_exact(mtx_view, tri_one, 0)
            """print '-'*80
            print 'mtx before right mul, after left'
            print mtx_view"""
            partial_right_one_exact(mtx_view, tri_one, 0)
            # print '-'*80
            # print 'mtx after right mul'
            # print mtx_view
            # print '-'*80
            mtx_view[1: m, 0] = 0
            mtx_view[0, 1: m] = 0
            # print '-'*80
            # print 'PL before right mul'
            # print PL
            # print '-'*80
            tri_one = -tri_one
            with bf.Context(precision=EXACT_PRECISION):
                for i in xrange(n):
                    ssum = bf.BigFloat(0)
                    for j in xrange(sum + 1, n, 1):
                        ssum = bf.add(ssum, bf.mul(np.float64(PL[i, j]), np.float64(tri_one[j - sum - 1])))

                    PL[i, sum] = bf.add(np.float64(PL[i, sum]), ssum)
                #PL[i, sum] += dot(PL[i, sum + 1:n], (-tri_one))
            # print '-'*80
            # print 'PL after right mul'
            # print PL
            # print '-'*80
        else:
            B_k_minus = -B_k

            tridiagonal[1, sum], tridiagonal[1, sum + 1] = T_k[0, 0], T_k[1, 1]
            tridiagonal[0, sum + 1] = T_k[0, 1]
            tridiagonal[2, sum] = T_k[1, 0]
            tri_one = np.zeros(m)
            tri_two = np.zeros(m)
            for i in xrange(m):
                tri_one[i] = bf.add(bf.mul(B_k_minus[i, 0], T_k_inverse[0, 0]), bf.mul(B_k_minus[i, 1], T_k_inverse[1, 0]))
                tri_two[i] = bf.add(bf.mul(B_k_minus[i, 0], T_k_inverse[0, 1]), bf.mul(B_k_minus[i, 1], T_k_inverse[1, 1]))
            #tri_one = triangular_view[2: m, 0]
            #tri_two = triangular_view[2: m, 1]
            """print '-'*80
            print 'mtx before left mul'
            print mtx_view"""
            partial_left_two_exact(mtx_view, tri_one, tri_two, 0, 1)
            """print '-'*80
            print 'mtx before right mul, after left'
            print mtx_view"""
            partial_right_two_exact(mtx_view, tri_one, tri_two, 0, 1)
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
            with bf.Context(precision=EXACT_PRECISION):
                tri_one = -tri_one
                tri_two = -tri_two
                for i in xrange(n):
                    ssum1 = bf.BigFloat(0)
                    ssum2 = bf.BigFloat(0)
                    for j in xrange(sum + 2, n, 1):
                        ssum1 = bf.add(ssum1, bf.mul(PL[i, j], tri_one[j - sum - 2]))
                        ssum2 = bf.add(ssum2, bf.mul(PL[i, j], tri_two[j - sum - 2]))

                    PL[i, sum] = bf.add(PL[i, sum], ssum1)
                    PL[i, sum + 1] = bf.add(PL[i, sum + 1], ssum2)
            """print '-'*80
            print 'PL after right mul'
            print PL
            print '-'*80"""
        sum += n_k
        cell_sizes.append(n_k)
        #flip = not flip
    #print 'ASSEMBLED:'
    #print dot(dot(PL, mtx), np.matrix(PL).getH())
    P, L = separate_permutation(PL, dtype=dtype)

    return P, L, cell_sizes, tridiagonal