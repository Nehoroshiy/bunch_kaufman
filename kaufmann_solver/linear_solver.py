import numpy as np
from numpy import float128 as f128, dot
from math import sqrt
from kaufmann_solver.bunch_kaufmann import bunch_kaufman
from kaufmann_solver.utils.iterative_refinement import conjugate_gradients_pract
from kaufmann_solver.utils.utils import euclid_vector_norm
from kaufmann_solver.utils.bunch_kaufman_utils import tridiagonal_inversion, tridiagonal_dot
from scipy import linalg, sparse


def symmetric_solve_simple(P, L, tridiagonal, cell_sizes, free_values, alpha=(1. + sqrt(17)) / 8):
    z = linalg.solve_triangular(L, P.T.dot(free_values), lower=True, unit_diagonal=True)
    #print tridiagonal
    w = linalg.solve_banded((1, 1), tridiagonal, z)
    #tri_inv = tridiagonal_inversion(tridiagonal, cell_sizes, dtype=f128)
    #w1 = tridiagonal_dot(tri_inv, z, dtype=f128)
    #print '-'*80
    #print 'difference between auto and manual tridiagonal solve:'
    #print euclid_vector_norm(w1 - w)
    #print '-'*80
    y = linalg.solve_triangular(np.matrix(L, dtype=f128).getH(), w, lower=False, unit_diagonal=True)
    return P.dot(y)


def isPSD(A, tol=1e-8):
  E,V = linalg.eigh(A)
  return np.all(E > -tol)


def symmetric_system_solve(system_matrix_origin, free_values_origin, alpha=(1. + sqrt(17)) / 8, precondition=False, regularize=False, refinement=False):
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
    n = system_matrix_origin.shape[0]
    if precondition and isPSD(system_matrix_origin):
        diag_l = np.zeros(n)
        diag_r = np.zeros(n)
        for i in xrange(n):
            diag_l[i] = 1.0 / euclid_vector_norm(system_matrix_origin[:, i])
            diag_r[i] = 1.0 / euclid_vector_norm(system_matrix_origin[i])
        mtx = np.array(system_matrix_origin, dtype=f128)
        free_values = np.array(free_values_origin, dtype=f128)
        for i in xrange(n):
            mtx[i] *= diag_l[i]
        for i in xrange(n):
            mtx[:, i] *= diag_r[i]
            free_values[i] *= diag_l[i]
        #free_values = dot(diag_l, free_values_origin)
        mtx = (mtx + mtx.T) / 2
        P, L, cell_sizes, tridiagonal = bunch_kaufman(mtx, alpha, regularize=regularize)
        computed_result = symmetric_solve_simple(P, L, tridiagonal, cell_sizes, free_values, alpha)
        for i in xrange(n):
            computed_result[i] *= diag_r[i]
    else:
        P, L, cell_sizes, tridiagonal = bunch_kaufman(system_matrix_origin, alpha, regularize=regularize)
        computed_result = symmetric_solve_simple(P, L, tridiagonal, cell_sizes, free_values_origin, alpha)
    if refinement:
        diags = [1,0,-1]
        T = sparse.spdiags(tridiagonal, diags, P.shape[0], P.shape[0], format='csc').todense()
        assembled_result = dot(dot(dot(dot(P, L), T), np.matrix(L).getH()), P.T)
        x1, k = conjugate_gradients_pract(np.array(assembled_result, dtype=f128), free_values, computed_result)
        computed_result = x1
    return computed_result