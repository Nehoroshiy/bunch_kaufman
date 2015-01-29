import numpy as np
from numpy import float128 as f128, dot
from math import sqrt
from kaufmann_solver.bunch_kaufmann import bunch_kaufman
from kaufmann_solver.utils.iterative_refinement import conjugate_gradients_pract, conjgrad
from kaufmann_solver.utils.utils import euclid_vector_norm, relative_error
from kaufmann_solver.utils.bunch_kaufman_utils import tridiagonal_inversion, tridiagonal_dot, tridiagonal_dot_exact, tridiagonal_inversion_exact
import tests.tests
from scipy import linalg, sparse


def symmetric_solve_simple(P, L, tridiagonal, cell_sizes, free_values, alpha=(1. + sqrt(17)) / 8):
    dtype=tridiagonal.dtype
    z = linalg.solve_triangular(L, P.T.dot(free_values), lower=True, unit_diagonal=True)
    #print tridiagonal
    w = linalg.solve_banded((1, 1), tridiagonal, z)
    tri_inv = tridiagonal_inversion(tridiagonal, cell_sizes, dtype=dtype)
    w1 = tridiagonal_dot(tri_inv, z, dtype=dtype)
    y = linalg.solve_triangular(np.matrix(L, dtype=dtype).getH(), w, lower=False, unit_diagonal=True)
    return P.dot(y)


def max_abs(v):
    return max(abs(v))


def linear_cholesky_solve(LD, P, free_values_original):
    n = LD.shape[0]
    free_values = np.array(free_values_original, dtype=f128)
    for (idx1, idx2) in P:
        free_values[[idx1, idx2]] = free_values[[idx2, idx1]]
    w = linalg.solve_triangular(np.tril(LD, -1) + np.identity(n), free_values, lower=True, unit_diagonal=True)
    diag_inverse = 1.0 / LD.diagonal()
    y = diag_inverse * w
    z = linalg.solve_triangular(np.triu(LD.T, 1) + np.identity(n), y, lower=False, unit_diagonal=True)
    for (idx1, idx2) in reversed(P):
        z[[idx1, idx2]] = z[[idx2, idx1]]
    return z


def symmetric_system_solve(system_matrix_origin, free_values_origin, alpha=(1. + sqrt(17)) / 8):
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
    dtype = system_matrix_origin.dtype
    n = system_matrix_origin.shape[0]
    free_values = free_values_origin
    P, L, cell_sizes, tridiagonal = bunch_kaufman(system_matrix_origin, alpha)
    computed_result = symmetric_solve_simple(P, L, tridiagonal, cell_sizes, free_values, alpha)
    return computed_result