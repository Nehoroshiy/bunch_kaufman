import numpy as np
from numpy import dot, float128 as f128, identity as I, tril
from math import factorial
from kaufmann_solver.utils.utils import frobenius_norm, euclid_vector_norm, max_pseudo_norm, relative_error
from kaufmann_solver.bunch_kaufmann import bunch_kaufman
from kaufmann_solver.linear_solver import symmetric_system_solve, linear_cholesky_solve
from kaufmann_solver.utils.bunch_kaufman_utils import exchange_rows, exchange_columns
from kaufmann_solver.cholesky import cholesky_diagonal
from scipy import sparse

from scipy.sparse.linalg import arpack

def isPSD(A, tol = 1e-8):
    vals, vecs = arpack.eigsh(A, k=2, which='BE') # return the ends of spectrum of A
    return np.all(vals > -tol)

# hilbert matrix (Hij = 1/(i + j - 1))

def boundline(n=80):
    print '-'*n

def binomial(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return factorial(n) // (factorial(k) * factorial(n-k))

def hilb(n, m=0):
    if n < 1 or m < 0:
        raise ValueError("Matrix size must be one or greater")
    elif n == 1 and (m == 0 or m == 1):
        return np.array([[1]])
    elif m == 0:
        m = n
    return 1. / (np.arange(1, n + 1) + np.arange(0, m)[:, np.newaxis])


def invhilb(n):
    H = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = ((-1)**(i + j)) * (i + j + 1) * binomial(n + i, n - j - 1) * \
            binomial(n + j, n - i - 1) * binomial(i + j, i) ** 2
    return H


def extended_factorization_test(mtx):
    res = open("res.txt", "w")
    for i in xrange(15, 50, 1):
        h = hilb(i)
        res.write('hilb size:')
        res.write(str(i))
        res.write('\t\t')
        P, L, cell_sizes, tridiagonal = bunch_kaufman(h.copy())
        if filter(lambda x: x != 1 and x != 2, cell_sizes):
            raise Exception('Cell sizes in Bunch-Kaufman must be 1-2')
        if not np.array_equal(L, np.tril(L)):
            raise Exception('Bunch-Kaufman algo must make lower triangular matrix')
        diags = [1,0,-1]
        T = np.array(sparse.spdiags(tridiagonal, diags, i, i, format='csc').todense(), dtype=np.float128)
        assembled_result = dot(dot(dot(dot(P, L), T), np.array(np.matrix(L).getH())), P.T)
        res.write(str(frobenius_norm(h - assembled_result)))
        res.write('\n')
    res.close()


def extended_linear_solve_hilbert_test(max_size=50):
    if max_size < 11:
        max_size = 11
    res = open("linear_res.txt", "w")
    for i in xrange(10, max_size, 1):
        test_solution = np.ones(i, dtype=f128)
        h = np.array(hilb(i), dtype=f128)
        free_values = dot(h, test_solution)
        res.write('hilb size:')
        res.write(str(i))
        res.write('\t\t')
        x_without_regularize = symmetric_system_solve(h, free_values, trusty=False)
        res.write(str(euclid_vector_norm(x_without_regularize - test_solution)))
        res.write('\t\t')
        computed_free_variables = dot(h, x_without_regularize)
        res.write(str(relative_error(free_values, computed_free_variables)))
        res.write('\t\t\t')
        x_with_regularize = symmetric_system_solve(h, free_values, trusty=True)
        res.write(str(euclid_vector_norm(x_with_regularize - test_solution)))
        res.write('\t\t')
        computed_free_variables = dot(h, x_with_regularize)
        res.write(str(relative_error(free_values, computed_free_variables)))
        res.write('\n')


def factorization_test(mtx, regularize=False):
    """Tests Bunch-Kaufman factorization for a given matrix.

        Test by factorizing it, restoring from factors and counting difference.

    Args:
         (np.array): testing matrix

    Returns:
        none

    Raises:
        Exception: An error occurred when Bunch-Kaufman doesn't work properly.

    """
    P, L, cell_sizes, tridiagonal = bunch_kaufman(mtx.copy(), regularize=regularize)
    if filter(lambda x: x != 1 and x != 2, cell_sizes):
        raise Exception('Cell sizes in Bunch-Kaufman must be 1-2')
    if not np.array_equal(L, np.tril(L)):
        raise Exception('Bunch-Kaufman algo must make lower triangular matrix')
    diags = [1,0,-1]
    T = sparse.spdiags(tridiagonal, diags, mtx.shape[0], mtx.shape[0], format='csc').todense()
    assembled_result = dot(dot(dot(dot(P, L), T), np.matrix(L).getH()), P.T)
    boundline()
    print 'This is Bunch-Kaufman test.'
    print 'Original matrix:'
    print np.matrix(mtx)
    boundline()
    print 'Assembled matrix'
    print np.matrix(assembled_result)
    boundline()
    print 'Factors:'
    print 'P:'
    print P
    boundline()
    print 'L:'
    print L
    boundline()
    print 'Tridiagonal:'
    print T
    boundline()
    print 'Frobenius norm of difference:'
    print frobenius_norm(mtx - assembled_result)
    print 'Maximum difference of elements:'
    print max_pseudo_norm(mtx - assembled_result)
    boundline()





def linear_cholesky(mtx, precondition=False):
    n = mtx.shape[0]
    mtx_precision = np.array(mtx, dtype=f128)
    original_solve_precision = np.zeros(mtx.shape[0], dtype=f128) + 1
    free_values_origin = dot(mtx_precision, original_solve_precision)
    """print 'Is matrix PSD?'
    #LD = np.linalg.cholesky(mtx)
    #P = []
    #for i in xrange(n):
    #    LD_view = LD[i:, i:]
    #    j = argmax(LD)
    print 'LD'
    print LD
    print '-'*80
    assembled_result = dot(LD, LD.T)
    print 'assembled result:'
    print assembled_result
    print 'matrix:'
    print mtx"""
    if precondition:
        diag_l = np.zeros(n)
        diag_r = np.zeros(n)
        for i in xrange(n):
            diag_l[i] = 1.0 / euclid_vector_norm(mtx_precision[:, i])
            diag_r[i] = 1.0 / euclid_vector_norm(mtx_precision[i])
        mtx = np.array(mtx_precision, dtype=f128)
        free_values = np.array(free_values_origin, dtype=f128)
        for i in xrange(n):
            mtx_precision[i] *= diag_l[i]
            free_values[i] *= diag_l[i]
        for i in xrange(n):
            mtx_precision[:, i] *= diag_r[i]
        #free_values = dot(diag_l, free_values_origin)
        mtx_precision = (mtx_precision + mtx_precision.T) / 2
        #tests.tests.factorization_test(mtx, False)
        LD, P = cholesky_diagonal(mtx_precision)
        computed_result = linear_cholesky_solve(LD, P, free_values)
        for i in xrange(n):
            computed_result[i] *= diag_r[i]
    else:
        free_values = free_values_origin
        LD, P = cholesky_diagonal(mtx_precision)
        computed_result = linear_cholesky_solve(LD, P, free_values)
    print 'original x:'
    print original_solve_precision
    print 'computed x:'
    print computed_result
    print '-'*80
    print 'free_values:'
    print free_values_origin
    print 'check free values:'
    check_values = dot(mtx_precision, computed_result / diag_r) / diag_l
    print check_values


def cholesky_test(mtx):
    n = mtx.shape[0]
    LD, P = cholesky_diagonal(mtx)
    D = np.zeros([n, n], dtype=f128)
    diag = LD.diagonal()
    np.fill_diagonal(D, diag)
    L = tril(LD, -1) + I(n)
    assembled_result = dot(dot(L, D), L.T)
    #assembled_result = dot(dot(dot(P, dot(L, D)), L.T), P.T)
    for (idx1, idx2) in reversed(P):
        exchange_rows(assembled_result, idx1, idx2)

    for (idx1, idx2) in reversed(P):
        exchange_columns(assembled_result, idx1, idx2)
    boundline()
    print 'This is Cholesky test.'
    print 'Original matrix:'
    print np.matrix(mtx)
    boundline()
    print 'Assembled matrix'
    print np.matrix(assembled_result)
    boundline()
    print 'Factors:'
    print 'P:'
    print P
    boundline()
    print 'L:'
    print tril(LD, -1) + I(n)
    boundline()
    print 'Diagonal:'
    print D
    boundline()
    print 'Frobenius norm of difference:'
    print frobenius_norm(mtx - assembled_result)
    print 'Maximum difference of elements:'
    print max_pseudo_norm(mtx - assembled_result)
    boundline()


def numpy_test(mtx, original_solve=[]):
    """Tests numpy linalg solution of a symmetric system.

    Args:
         (np.array): testing matrix
         (np.array): original solve for testing

    Returns:
        none

    Raises:
        Exception: An error occurred when sizes of matrix and original solution doesn't fit.

    """
    if not original_solve:
        original_solve = np.zeros(mtx.shape[0]) + 1
        original_solve_precision = np.zeros(mtx.shape[0], dtype=np.float128) + 1
    if original_solve.shape[0] != mtx.shape[0]:
        raise Exception('Sizes of matrix and original solve must be equal!')
    mtx_precision = np.array(mtx, dtype=np.float128)
    free_variables = mtx.dot(original_solve)
    free_variables_precision = mtx_precision.dot(original_solve_precision)

    result = np.linalg.solve(mtx, np.array(free_variables_precision, dtype=np.float64))

    boundline()
    print 'This is numpy linear symmetric system solver test.'
    print 'Original result:'
    print original_solve
    print 'Numpy result:'
    print result

    print 'Euclid norm of delta:'
    print euclid_vector_norm(result - original_solve)
    boundline()


def linear_solve_test(mtx, precondition=False, regularize=False):
    """Tests Bunch-Kaufman-based symmetric system solver for a given matrix.

    Args:
         (np.array): testing matrix
         (np.array): original solve for testing

    Returns:
        none

    Raises:
        Exception: An error occurred when sizes of matrix and original solution doesn't fit.

    """

    original_solve = np.zeros(mtx.shape[0]) + 1
    original_solve_precision = np.zeros(mtx.shape[0], dtype=np.float128) + 1
    if original_solve.shape[0] != mtx.shape[0]:
        raise Exception('Sizes of matrix and original solve must be equal!')
    mtx_precision = np.array(mtx, dtype=np.float128)
    free_variables = mtx.dot(original_solve)
    free_variables_precision = mtx_precision.dot(original_solve_precision)

    kaufmann_result = symmetric_system_solve(mtx, np.array(free_variables_precision, dtype=np.float64), precondition=precondition, regularize=regularize)

    boundline()
    print 'This is linear symmetric system solver test.'
    print 'Original free variables:'
    print free_variables
    print 'Precise free variables:'
    print free_variables_precision
    print 'euclid norm of difference:'
    print euclid_vector_norm(free_variables - free_variables_precision)
    print 'Counted free variables:'
    count_free = dot(mtx_precision, kaufmann_result)
    print count_free
    print 'difference:'
    print euclid_vector_norm(free_variables_precision - count_free)
    boundline()
    print 'Original result:'
    print original_solve
    print 'Kaufmann result:'
    print kaufmann_result

    print 'Euclid norm of delta:'
    print euclid_vector_norm(kaufmann_result - original_solve)
    boundline()