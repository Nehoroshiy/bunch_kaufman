import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
from scipy import linalg, sparse
from numpy import identity as I, dot
from kaufmann_solver.utils.utils import transposition_matrix, separate_permutation, triangular_inversion, \
    euclid_vector_norm, relative_error
from operator import itemgetter
from math import sqrt
from numpy import float128 as f128



def inverse_1_2(small_matrix, dtype=np.float64):
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
    """


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


def bunch_kaufman_copy(mtx_origin, alpha=(1. + sqrt(17)) / 8, regular_coefficient=1e-4):
    mtx = np.array(mtx_origin, dtype=f128)
    n = mtx.shape[0]
    tridiagonal = np.zeros([3, n], dtype=f128)
    sum = 0
    cell_sizes = []
    PL = I(n, dtype=f128)
    while sum < n:
        m = n - sum
        mtx_view = mtx[sum: n, sum: n]
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
            exchange_columns(PL, sum, sum + j_idx)
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
            #print mtx_view
            partial_left_one(mtx_view, tri_one, 0)
            #print '-'*80
            #print mtx_view
            #print '-'*80
            partial_right_one(mtx_view, tri_one, 0)
            #print mtx_view
            #print '!'*80
            mtx_view[1: m, 0] = 0
            mtx_view[0, 1: m] = 0
            #print mtx_view
            #print '-'*80
            #partial_left_one(PL_view, -tri_one, 0)
            for i in xrange(n):
                PL[i, sum] += dot(PL[i, sum + 1:n], (-tri_one))
            #for i in xrange(m):
            #    PL_view[i, 0] += dot(PL_view[i, 1: m], (-tri_one))
            #partial_right_one(PL_view, -tri_one, 0)
            #print PL_view
            #print '-'*80
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
            #partial_left_two(PL_view, -tri_one, -tri_two, 0, 1)
            #partial_right_two(PL_view, -tri_one, -tri_two, 0, 1)
            for i in xrange(n):
                PL[i, sum] += dot(PL[i, sum + 2:n], (-tri_one))
                PL[i, sum + 1] += dot(PL[i, sum + 2:n], (-tri_two))

        #mtx_view[:, :] = dot(dot(triangular_view, mtx_view), np.matrix(triangular_view).getH())[:, :]
        #PL = dot(PL, -triangular + 2*I(n))
        #print PL
        """for (i, j) in zip(*mtx_view.nonzero()):
            if mtx_view[i, j] < regular_coefficient:
                mtx_view[i, j] = mtx_view[j, i] = regular_coefficient * (1.0 + np.random.random())"""
        sum += n_k
        cell_sizes.append(n_k)
    P, L = separate_permutation(PL, dtype=f128)
    #diags = [-1, 0, 1]
    #A = sparse.spdiags(tridiagonal, diags, n, n, format='csc')
    #print A.todense()
    #print '-'*80
    #print mtx
    regular_coefficient = 1e-4
    regular_coefficient2 = 1e-4 + 1e-5
    from random import random
    for w, (i, j) in enumerate(zip(*tridiagonal.nonzero())):
        if tridiagonal[i, j] < regular_coefficient:
            tridiagonal[i, j] = regular_coefficient if w % 2 else regular_coefficient2
            #tridiagonal[i, j] *= 10*random()
    for i in xrange(n):
        if tridiagonal[1, i] == 0:
            tridiagonal[1, i] = regular_coefficient
    #for i in xrange(n):
    #    tridiagonal[1, i] *= 1.5

    return mtx, P, L, cell_sizes, tridiagonal


def bunch_kaufmann(mtx_origin, alpha=(1. + sqrt(17)) / 8):
    mtx = mtx_origin.copy()
    n = mtx.shape[0]
    sum = 0
    cell_sizes = []
    PL = I(n)
    while sum < n:
        mtxs = mtx[sum: n, sum: n]
        idx = max([(abs(mtxs[j][j]), j) for j in xrange(mtxs.shape[0])], key=itemgetter(0))[1]

        permutation_step = transposition_matrix(n, sum, sum + idx)
        permutation = permutation_step[sum: n, sum: n]
        triangular_step = I(n)
        triangular = triangular_step[sum: n, sum: n]
        # conjugate M' with permutation matrix
        mtxs[:, :] = dot(dot(permutation, mtxs), permutation.T)[:, :]

        PL = dot(PL, permutation_step)
        # find index for larger column abs and this abs
        [lambda_val, idx] = max([(abs(mtxs[j][0]), j) for j in xrange(mtxs.shape[0])], key=itemgetter(0))
        if abs(mtxs[0][0]) >= alpha * lambda_val:
            n_k = 1
            if mtx.shape[0] <= sum + n_k:
                cell_sizes.append(n_k)
                break
            permutation[:, :] = I(mtxs.shape[0])[:, :]
        else:
            [sigma_val, j_idx] = max([(abs(mtxs[j][idx]), j) for j in xrange(mtxs.shape[0]) if j != idx],
                                     key=itemgetter(0))
            if sigma_val * abs(mtxs[0][0]) >= alpha * lambda_val ** 2:
                n_k = 1
                if mtx.shape[0] <= sum + n_k:
                    cell_sizes.append(n_k)
                    break
                permutation[:, :] = I(mtxs.shape[0])[:, :]
            else:
                if abs(mtxs[idx][idx]) >= alpha * sigma_val:
                    n_k = 1
                    if mtx.shape[0] <= sum + n_k:
                        cell_sizes.append(n_k)
                        break
                    permutation[:, :] = transposition_matrix(mtxs.shape[0], 0, idx)[:, :]

                else:
                    n_k = 2
                    if mtx.shape[0] <= sum + n_k:
                        cell_sizes.append(n_k)
                        break
                    permutation[:, :] = dot(transposition_matrix(mtxs.shape[0], 1, idx),
                                            transposition_matrix(mtxs.shape[0], 0, j_idx))[:, :]
        mtxs_image = np.dot(np.dot(permutation, mtxs), permutation.T)
        T_k = mtxs_image[0:n_k, 0:n_k]
        T_k_inverse = inverse_1_2(T_k)
        B_k = mtxs_image[n_k: mtxs_image.shape[0], 0: n_k]
        triangular[n_k:triangular.shape[0], 0:n_k] = -B_k.dot(T_k_inverse)

        triangular_vectors = np.tril(triangular, -1)[:, 0:n_k].T

        #mtxs[:, :] = dot(dot(permutation, mtxs), permutation.T)
        #partial_product_left(mtxs, triangular_vectors, range(n_k))
        #partial_product_right(mtxs, triangular_vectors, range(n_k))

        mtxs[:, :] = dot(dot(dot(dot(triangular, permutation), mtxs), permutation.T), np.matrix(triangular).getH())[:,:]
        # For gaussian atomic matrix M inverse(M) = -M + 2I, I is identity matrix
        #PL = dot(PL, permutation_step.T)
        #triangular_vectors = np.tril((-triangular_step + 2*I(n))[sum:n, sum:n], -1)[:, 0:n_k].T
        #partial_product_right(PL[sum: n, sum: n], triangular_vectors, range(n_k))
        PL = dot(dot(PL, permutation_step.T), -triangular_step + 2*I(n))
        sum += n_k
        cell_sizes.append(n_k)
    P, L = separate_permutation(PL)
    #print P
    #print '-'*80
    #print L
    return mtx, P, L, cell_sizes

def symmetric_system_solve_without_refinement(P, L, tridiagonal, free_values, alpha=(1. + sqrt(17)) / 8):
    """Solves linear system with Bunch-Kaufman factorization.

        To solve system Ax = b, we need to solve next systems:
        Lz = P^t b,
        Tw = z,
        L^* y = w,
        x = Py

    Args:
        system_matrix_origin (np.array): matrix of linear system
        free_values (np.array): vector of free values
        alpha (float): tuning coefficient for Bunch-Kaufmann algorithm

    Returns:
        np.array: solution of system

    Raises:
        Exception: An error occurred while passing non-square matrix
        Exception: An error occurred while passing non-triangular matrix
        Exception: An error occurred while passing singular matrix
    """
    z = np.linalg.solve(L, P.T.dot(free_values))
    w = np.linalg.solve(tridiagonal, z)
    y = np.linalg.solve(np.matrix(L).getH(), w)
    return P.dot(y)


def symmetric_solve_simple(P, L, tridiagonal, cell_sizes, free_values, alpha=(1. + sqrt(17)) / 8):
    z = linalg.solve_triangular(L, P.T.dot(free_values), lower=True, unit_diagonal=True)
    print tridiagonal
    w = linalg.solve_banded((1, 1), tridiagonal, z)
    #tri_inv = tridiagonal_inversion(tridiagonal, cell_sizes, dtype=f128)
    #w1 = tridiagonal_dot(tri_inv, z, dtype=f128)

    """print '-'*80
    print 'w:'
    print w
    print 'w1:'
    print w1
    print 'HOW ABOUT DELTA?'
    print euclid_vector_norm(w - w1)
    print '-'*80"""
    y = linalg.solve_triangular(np.matrix(L, dtype=f128).getH(), w, lower=False, unit_diagonal=True)
    return P.dot(y)


def tridiagonal_dot(tridiagonal, v, dtype=np.float64):
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


def symmetric_system_solve(system_matrix_origin, free_values, alpha=(1. + sqrt(17)) / 8, eps=1e-16):
    """Solves linear system with Bunch-Kaufman factorization and simple error correction.

        We solve system Ax = b, and take x_computed. Then, we find err = (b - A*x_computed),
        make error correction x_computed += err, and iterates while err > EPSILON

    Args:
        system_matrix_origin (np.array): matrix of linear system
        free_values (np.array): vector of free values
        alpha (float): tuning coefficient for Bunch-Kaufman algorithm
        eps (float): precision factor of an iterative refinement

    Returns:
        np.array: solution of system

    Raises:
        Exception: An error occurred while passing non-square matrix
        Exception: An error occurred while passing non-triangular matrix
        Exception: An error occurred while passing singular matrix
    """

    mtx, P, L, cell_sizes, tridiagonal = bunch_kaufman_copy(system_matrix_origin, alpha)
    computed_result = symmetric_solve_simple(P, L, tridiagonal, cell_sizes, free_values, alpha)
    diags = [1,0,-1]
    T = sparse.spdiags(tridiagonal, diags, mtx.shape[0], mtx.shape[0], format='csc').todense()
    assembled_result = dot(dot(dot(dot(P, L), T), np.matrix(L).getH()), P.T)
    x1, k = conjugate_gradients_pract(np.array(assembled_result, dtype=f128), free_values, computed_result)
    print '-'*80
    print 'result:'
    print computed_result
    print 'and with refinement:'
    #print x1
    print 'HOW ABOUT DELTA?'
    #print euclid_vector_norm(computed_result - x1)
    print '-'*80
    return x1
    """mtx, P, L, cell_sizes, tridiagonal = bunch_kaufman_copy(system_matrix_origin, alpha)
    computed_result = symmetric_solve_simple(P, L, tridiagonal, cell_sizes, free_values, alpha)
    residual = dot(system_matrix_origin, computed_result) - free_values
    while euclid_vector_norm(residual) >= eps:
        mu = dot(residual, residual) / dot(dot(residual, system_matrix_origin), residual)
        computed_result -= mu * residual
        residual = dot(system_matrix_origin, computed_result) - free_values
    return computed_result"""
    """mtx, P, L, cell_sizes, tridiagonal = bunch_kaufman_copy(system_matrix_origin, alpha)
    computed_result = symmetric_solve_simple(P, L, tridiagonal, free_values, alpha)
    residual = free_values - dot(system_matrix_origin, computed_result)
    while euclid_vector_norm(residual) >= eps:
        computed_result += symmetric_solve_simple(P, L, tridiagonal, residual, alpha)
        residual = free_values - dot(system_matrix_origin, computed_result)
    return computed_result
    """


def symmetric_system_solve_old(system_matrix_origin, free_values, alpha=(1. + sqrt(17)) / 8, eps=1e-15):
    """Solves linear system with Bunch-Kaufman factorization and simple error correction.

        We solve system Ax = b, and take x_computed. Then, we find err = (b - A*x_computed),
        make error correction x_computed += err, and iterates while err > EPSILON

    Args:
        system_matrix_origin (np.array): matrix of linear system
        free_values (np.array): vector of free values
        alpha (float): tuning coefficient for Bunch-Kaufman algorithm
        eps (float): precision factor of an iterative refinement

    Returns:
        np.array: solution of system

    Raises:
        Exception: An error occurred while passing non-square matrix
        Exception: An error occurred while passing non-triangular matrix
        Exception: An error occurred while passing singular matrix
    """
    mtx, P, L, cell_sizes = bunch_kaufmann(system_matrix_origin, alpha)
    computed_result = symmetric_system_solve_without_refinement(P, L, mtx, free_values, alpha)
    """residual = free_values - dot(system_matrix_origin, computed_result)
    prev_norm = euclid_vector_norm(residual)
    while euclid_vector_norm(residual) >= eps:
        computed_result += symmetric_system_solve_without_refinement(P, L, mtx, residual, alpha)
        residual = free_values - dot(system_matrix_origin, computed_result)
        if prev_norm < euclid_vector_norm(residual):
            break"""
    return computed_result


def conjugate_gradients(A, b, x0, tolerance=1e-16):
    k = 0
    residual = b - dot(A, x0)
    computed_result = x0
    q = 0
    q_prev = 0
    c = 0
    norm = euclid_vector_norm(residual)
    while norm > tolerance:
        q_prev = q
        q = residual / norm
        k += 1
        a_step = dot(dot(q, A), q)
        if k == 1:
            d = a_step
            v = norm / d
            c = q
        else:
            l = norm / d
            d = a_step - norm * l
            v = - norm * v / d
            c = q - l * c
        computed_result += v * c
        residual = dot(A, q) - a_step * q - norm * q_prev
        norm = euclid_vector_norm(residual)
    return computed_result


def conjugate_gradients_pract(A, b, x0, tolerance=1e-17):
    k = 0
    x = x0.copy()
    r = b - dot(A, x)
    ro_c = dot(r, r)
    delta = tolerance * euclid_vector_norm(b)
    p = 0
    ro_m = 0
    while sqrt(ro_c) >= delta:
        k += 1
        if k == 1:
            p = r
        else:
            tau = ro_c / ro_m
            p = r + tau * p
        w = dot(A, p)
        mu = ro_c / dot(p, w)
        x += mu * p
        r -= mu * w
        ro_m = ro_c
        ro_c = dot(r, r)
        if k > 50000:
            break

    return x, k


def preconditioned_conjugate_gradients_diag(A, b, x0, tolerance=1e-17):
    M = A.diagonal()
    k = 0
    r = b - dot(A, x0)
    x = x0.copy()
    z = M * r
    z_prev = 0
    r_prev = 0
    while euclid_vector_norm(r) > tolerance:
        k += 1
        if k == 1:
            p = z
        else:
            tau = dot(r, z) / dot(r_prev, z_prev)
            p = z + tau * p
        mu = dot(r, z) / dot(dot(p, A), p)
        x -= mu * p
        r_prev = r
        r -= mu * dot(A, p)
        z_prev = z
        z = M * r
    return x, k