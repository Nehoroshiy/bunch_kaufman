import numpy as np
from numpy import identity as I, dot
from kaufmann_solver.utils.utils import transposition_matrix, separate_permutation, triangular_inversion, \
    euclid_vector_norm, relative_error
from operator import itemgetter
from math import sqrt


def inverse_1_2(small_matrix):
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
        return np.array([[1.0 / small_matrix[0, 0]]])
    if small_matrix.shape != (2, 2):
        raise Exception("Matrix isn't 2x2 matrix")
    a = small_matrix[0, 0]
    b = small_matrix[0, 1]
    c = small_matrix[1, 0]
    d = small_matrix[1, 1]
    det = a * d - b * c
    if det == 0:
        raise Exception('Matrix of size 2x2 is singular')
    inverse = np.zeros(small_matrix.shape)
    inverse[0, 0] = d / det
    inverse[0, 1] = -c / det
    inverse[1, 0] = -b / det
    inverse[1, 1] = a / det
    return inverse


def bunch_kaufmann(mtx_origin, alpha=(1. + sqrt(17)) / 8):
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

        mtxs[:, :] = dot(dot(dot(dot(triangular, permutation), mtxs), permutation.T), np.matrix(triangular).getH())[:,:]
        # For gaussian atomic matrix M inverse(M) = -M + 2I, I is identity matrix
        PL = dot(dot(PL, permutation_step.T), -triangular_step + 2*I(n))
        sum += n_k
        cell_sizes.append(n_k)
    P, L = separate_permutation(PL)
    #print P
    #print '-'*80
    #print L
    return mtx, P, L, cell_sizes


def symmetric_system_solve_without_refinement(system_matrix_origin, free_values, alpha=(1. + sqrt(17)) / 8):
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
    system_matrix = system_matrix_origin.copy()
    tridiagonal, P, L, cell_sizes = bunch_kaufmann(system_matrix, alpha)
    z = np.linalg.solve(L, P.T.dot(free_values))
    w = np.linalg.solve(tridiagonal, z)
    y = np.linalg.solve(np.matrix(L).getH(), w)
    return P.dot(y)


def symmetric_system_solve(system_matrix_origin, free_values, alpha=(1. + sqrt(17)) / 8, eps=1e-9):
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
    computed_result = symmetric_system_solve_without_refinement(system_matrix_origin, free_values, alpha)
    residual = free_values - dot(system_matrix_origin, computed_result)
    while euclid_vector_norm(residual) >= eps:
        computed_result += symmetric_system_solve_without_refinement(system_matrix_origin, residual, alpha)
        residual = free_values - dot(system_matrix_origin, computed_result)
    return computed_result