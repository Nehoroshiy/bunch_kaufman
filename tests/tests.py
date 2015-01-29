import numpy as np
from numpy import dot, float128 as f128, identity as I, tril
from math import factorial
from kaufmann_solver.utils.utils import frobenius_norm, euclid_vector_norm, max_pseudo_norm, relative_error
from kaufmann_solver.bunch_kaufmann import bunch_kaufman
from kaufmann_solver.linear_solver import symmetric_system_solve, linear_cholesky_solve
from kaufmann_solver.utils.bunch_kaufman_utils import exchange_rows, exchange_columns
from kaufmann_solver.cholesky import cholesky_diagonal
from scipy import sparse
import re
import pandas as pd

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

def hilb(n, m=0, dtype=np.float64):
    if n < 1 or m < 0:
        raise ValueError("Matrix size must be one or greater")
    elif n == 1 and (m == 0 or m == 1):
        return np.array([[1]])
    elif m == 0:
        m = n
    return np.array((1. / (np.arange(1, n + 1) + np.arange(0, m)[:, np.newaxis])), dtype=dtype)


def invhilb(n):
    H = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = ((-1)**(i + j)) * (i + j + 1) * binomial(n + i, n - j - 1) * \
            binomial(n + j, n - i - 1) * binomial(i + j, i) ** 2
    return H


def bunch_factorization_test(size):
    h = hilb(size)
    for i in xrange(50):
        bunch_kaufman(h.copy())


def cholesky_factorization_test(size):
    h = hilb(size)
    for i in xrange(50):
        cholesky_diagonal(h.copy())


def extended_linear_solve_hilbert_test(max_size=50, step=1, dtype=np.float64):
    dtype=np.array([1], dtype=dtype).dtype
    if max_size < 11:
        max_size = 11
    res = open("linear_hilb_bunch_" + str(dtype) + ".txt", "w")
    res.write('HILB DIM:\t||A - PLTL^tP^t||\tdx/x[' + str(dtype) + ']\tdb/b[' + str(dtype) + ']\n')
    for i in xrange(10, max_size + 1, step):
        print 'bk step:', i
        test_solution = np.ones(i, dtype=dtype)
        h = np.array(hilb(i), dtype=dtype)

        free_values = dot(h, test_solution)
        res.write(str(i))

        res.write('\t')

        P, L, cell_sizes, tridiagonal = bunch_kaufman(h)
        diags = [1,0,-1]
        T = np.array(sparse.spdiags(tridiagonal, diags, i, i, format='csc').todense(), dtype=np.float128)
        assembled_result = dot(dot(dot(dot(P, L), T), np.array(np.matrix(L).getH())), P.T)
        res.write(str(frobenius_norm(h - assembled_result)))

        res.write('\t')

        x = symmetric_system_solve(h, free_values)
        res.write(str(relative_error(test_solution, x)))

        res.write('\t')

        computed_free_variables = dot(h, x)
        res.write(str(relative_error(free_values, computed_free_variables)))
        res.write('\n')


def end_matcher(min_index):
    return re.compile('^\d+:' + str(min_index) + ':(\d+):*')

def begin_matcher(max_index):
    return re.compile('^\d+:(\d+):' + str(max_index) + ':.*')


def get_all_repeats(min_size, max_size, method='bunch', matrix_type='random', dtype=np.float128):
    import os
    import fnmatch
    os.chdir('many_randoms')
    dtype = np.array([1], dtype=dtype).dtype
    pos='positive'
    unfull_name_rare = ':'.join([str(min_size), str(max_size), matrix_type, pos, method, str(dtype)]) + '.txt'

    #lst = [file for file in os.listdir('.') if fnmatch.fnmatch(file, '*:' + unfull_name_rare)]
    b_m = begin_matcher(max_size)
    lst = sorted([file for file in os.listdir('.') if b_m.match(file) and file[file.index(':') + 1:] == unfull_name_rare], key=lambda x: int(x[:x.index(':')]))
    #print lst
    current_index = int(b_m.findall(lst[0])[0])
    indices = set([int(file[:file.index(':')]) for file in lst])
    big_lst = [lst]
    while current_index != min_size:
        b_m = begin_matcher(current_index - 1)
        big_lst.append(sorted([file for file in os.listdir('.') if b_m.match(file) and int(file[:file.index(':') and file[file.index(':') + 1:] == unfull_name_rare]) in indices], key=lambda x: int(x[:x.index(':')])))
        current_index = int(b_m.findall(big_lst[-1][0])[0])
        pds = [pd.read_csv(file, sep='\t', header=0, dtype={'a': int, 'b': f128, 'c': f128, 'd': f128, 'e': f128}) for file in big_lst[-1]]
    for lt in reversed(big_lst[:-1]):
        for i, df in enumerate(pds):
            #print 'LT[i]'
            #print lt[i]
            #print '-'*80
            #print pds[i]
            #print '-'*80
            pds[i] = pds[i].append(pd.read_csv(lt[i], sep='\t', header=0, dtype={'a': int, 'b': f128, 'c': f128, 'd': f128, 'e': f128}))
            #print pds[i]
            #print '-'*80
    #pds = [pd.read_csv(file, sep='\t', header=0, dtype={'a': int, 'b': f128, 'c': f128, 'd': f128, 'e': f128}) for file in lst]
    os.chdir('..')
    return pds


def extended_ckhol_linear_solve_hilbert_test(max_size=50, step=1, dtype=np.float64):
    dtype=np.array([1], dtype=dtype).dtype
    if max_size < 11:
        max_size = 11
    res = open("linear_hilb_chol_" + str(dtype) + ".txt", "w")
    res.write('HILB DIM:\t||A - PLDL^tP^t||\tdx/x[' + str(dtype) + ']\tdb/b[' + str(dtype) + ']\n')
    for i in xrange(10, max_size + 1, step):
        print 'chol step:', i
        test_solution = np.ones(i, dtype=dtype)
        h = np.array(hilb(i), dtype=dtype)

        free_values = dot(h, test_solution)
        res.write(str(i))

        res.write('\t')

        LD, P = cholesky_diagonal(h)
        D = np.zeros([i, i], dtype=f128)
        diag = LD.diagonal()
        np.fill_diagonal(D, diag)
        L = tril(LD, -1) + I(i)
        assembled_result = dot(dot(L, D), L.T)
        for (idx1, idx2) in reversed(P):
            exchange_rows(assembled_result, idx1, idx2)

        for (idx1, idx2) in reversed(P):
            exchange_columns(assembled_result, idx1, idx2)
        res.write(str(frobenius_norm(h - assembled_result)))

        res.write('\t')

        x = linear_cholesky_solve(LD, P, free_values)
        res.write(str(relative_error(test_solution, x)))

        res.write('\t')

        computed_free_variables = dot(h, x)
        res.write(str(relative_error(free_values, computed_free_variables)))
        res.write('\n')


def extended_linear_solve_random_test(max_size=50, step=1, dtype=np.float64, avg=10, positive_def=False):
    dtype=np.array([1], dtype=dtype).dtype
    if max_size < 11:
        max_size = 11
    if positive_def:
        res = open(str(max_size) + "_linear_random_positive_bunch_" + str(dtype) + ".txt", "w")
    else:
        res = open(str(max_size) + "_linear_random_bunch_" + str(dtype) + ".txt", "w")
    res.write('RAND DIM:\tCOND MIN:\tCOND AVG:\tCOND MAX\t||A - PLTL^tP^t|| MIN\t||A - PLTL^tP^t|| AVG\t||A - PLTL^tP^t|| MAX\tdx/x[' + str(dtype) + ']MIN\tdx/x[' + str(dtype) + ']AVG\tdx/x[' + str(dtype) + ']MAX\tdb/b[' + str(dtype) + ']MIN\tdb/b[' + str(dtype) + ']AVG\tdb/b[' + str(dtype) + ']MAX\n')
    for i in xrange(10, max_size + 1, step):
        print 'step:', i
        cond_avg = 0
        cond_max = 0
        cond_min = 1e40
        factor_difference_min = 1e15
        factor_difference_avg = 0
        factor_difference_max = 0
        relative_error_min = 1e15
        relative_error_avg = 0
        relative_error_max = 0
        residual_min = 1e15
        residual_avg = 0
        residual_max = 0
        for j in xrange(avg):
            test_solution = np.ones(i, dtype=dtype)
            h = np.random.randn(i, i)
            if positive_def:
                h = dot(h.T, h)
            h = (h + h.T) / 2
            cc = np.linalg.cond(h)
            h = np.array(h, dtype=f128)
            cond_avg += cc
            if cc > cond_max: cond_max = cc
            if cc < cond_min: cond_min = cc

            free_values = dot(h, test_solution)

            P, L, cell_sizes, tridiagonal = bunch_kaufman(h)
            diags = [1,0,-1]
            T = np.array(sparse.spdiags(tridiagonal, diags, i, i, format='csc').todense(), dtype=np.float128)
            assembled_result = dot(dot(dot(dot(P, L), T), np.array(np.matrix(L).getH())), P.T)
            cdiff = frobenius_norm(h - assembled_result)
            factor_difference_avg += cdiff
            if cdiff > factor_difference_max: factor_difference_max = cdiff
            if cdiff < factor_difference_min: factor_difference_min = cdiff

            x = symmetric_system_solve(h, free_values)
            rel_err = relative_error(test_solution, x)
            relative_error_avg += rel_err
            if rel_err > relative_error_max: relative_error_max = rel_err
            if rel_err < relative_error_min: relative_error_min = rel_err

            computed_free_variables = dot(h, x)
            cresidual = relative_error(free_values, computed_free_variables)
            residual_avg += cresidual
            if cresidual > residual_max: residual_max = cresidual
            if cresidual < residual_min:residual_min = cresidual
        cond_avg /= avg
        factor_difference_avg /= avg
        relative_error_avg /= avg
        residual_avg /= avg

        res.write(str(i))
        res.write('\t')
        res.write(str(cond_min) + '\t' + str(cond_avg) + '\t' + str(cond_max) + '\t')
        res.write(str(factor_difference_min) + '\t' + str(factor_difference_avg) + '\t' + str(factor_difference_max) + '\t')
        res.write(str(relative_error_min) + '\t' + str(relative_error_avg) + '\t' + str(relative_error_avg) + '\t')
        res.write(str(residual_min) + '\t' + str(residual_avg) + '\t' + str(residual_avg) + '\n')


def many_randoms(min_size=10, max_size=50, repeats=50, dtype=np.float64, positive_def=True):
    dtype=np.array([1], dtype=dtype).dtype
    import os
    import fnmatch
    os.chdir('many_randoms')
    mask_bunch = 'random:positive:bunch'
    mask_chol = 'random:positive:chol'
    unfull_bunch = ':'.join([str(min_size), str(max_size), mask_bunch, str(dtype)]) + '.txt'
    unfull_chol = ':'.join([str(min_size), str(max_size), mask_chol, str(dtype)]) + '.txt'
    lst = [int(file[:file.index(':')]) for file in os.listdir('.') if fnmatch.fnmatch(file, '*:' + unfull_bunch)]
    start = max(lst) if lst else -1
    f_bunch = open(str(start + repeats) + ':' + unfull_bunch, 'w')
    f_chol = open(str(start + repeats) + ':' + unfull_chol, 'w')
    f_bunch.close()
    f_chol.close()
    if max_size < 10:
        max_size = 10

    for N in xrange(start + 1, start + repeats + 1, 1):
        if positive_def:
            res_1 = open(str(N) + ':' + str(min_size) + ':' + str(max_size) + ":random:positive:bunch:" + str(dtype) + ".txt", "w")
            res_2 = open(str(N) + ':' + str(min_size) + ':' + str(max_size) + ":random:positive:chol:" + str(dtype) + ".txt", "w")
        else:
            res_1 = open(str(N) + ':' + str(min_size) + ':' + str(max_size) + ":random:bunch:" + str(dtype) + ".txt", "w")
            res_2 = open(str(N) + ':' + str(min_size) + ':' + str(max_size) + ":random:chol:" + str(dtype) + ".txt", "w")
        res_1.write('n\tcond\tdelta\tdx\tdb\n')
        res_2.write('n\tcond\tdelta\tdx\tdb\n')
        for i in xrange(min_size, max_size+1, 1):
            print 'filenum:', N,  '\tstep:', i
            test_solution = np.ones(i, dtype=dtype)
            h = np.random.randn(i, i)
            if positive_def:
                h = dot(h.T, h)
            h = (h + h.T) / 2
            cond = np.linalg.cond(h)
            h = np.array(h, dtype=f128)

            free_values = dot(h, test_solution)

            P, L, cell_sizes, tridiagonal = bunch_kaufman(h)
            diags = [1,0,-1]
            T = np.array(sparse.spdiags(tridiagonal, diags, i, i, format='csc').todense(), dtype=np.float128)
            assembled_result = dot(dot(dot(dot(P, L), T), np.array(np.matrix(L).getH())), P.T)
            factor_difference_1 = frobenius_norm(h - assembled_result)

            x = symmetric_system_solve(h, free_values)
            rel_err_1 = relative_error(test_solution, x)

            computed_free_variables = dot(h, x)
            residual_1 = relative_error(free_values, computed_free_variables)

            LD, P = cholesky_diagonal(h)
            D = np.zeros([i, i], dtype=f128)
            diag = LD.diagonal()
            np.fill_diagonal(D, diag)
            L = tril(LD, -1) + I(i)
            assembled_result = dot(dot(L, D), L.T)
            for (idx1, idx2) in reversed(P):
                exchange_rows(assembled_result, idx1, idx2)

            for (idx1, idx2) in reversed(P):
                exchange_columns(assembled_result, idx1, idx2)
            factor_difference_2 = frobenius_norm(h - assembled_result)

            x = linear_cholesky_solve(LD, P, free_values)
            rel_err_2 = relative_error(test_solution, x)

            computed_free_variables = dot(h, x)
            residual_2 = relative_error(free_values, computed_free_variables)

            res_1.write(str(i) + '\t')
            res_1.write(str(cond) + '\t')
            res_1.write(str(factor_difference_1) + '\t')
            res_1.write(str(rel_err_1) + '\t')
            res_1.write(str(residual_1) + '\n')
            res_2.write(str(i) + '\t')
            res_2.write(str(cond) + '\t')
            res_2.write(str(factor_difference_2) + '\t')
            res_2.write(str(rel_err_2) + '\t')
            res_2.write(str(residual_2) + '\n')
        res_1.close()
        res_2.close()


def fast_empiric_random_test(min_size=10, max_size=50, step=1, dtype=np.float64, positive_def=False):
    dtype=np.array([1], dtype=dtype).dtype
    if max_size < 11:
        max_size = 11
    if positive_def:
        res_1 = open(str(min_size) + '_' + str(max_size) + "_empiric_random_positive_bunch_" + str(dtype) + ".txt", "w")
        res_2 = open(str(min_size) + '_' + str(max_size) + "_empiric_random_positive_chol_" + str(dtype) + ".txt", "w")
    else:
        res_1 = open(str(min_size) + '_' + str(max_size) + "_empiric_random_bunch_" + str(dtype) + ".txt", "w")
        res_2 = open(str(min_size) + '_' + str(max_size) + "_empiric_random_chol_" + str(dtype) + ".txt", "w")
    res_1.write('RAND DIM:\tCOND:\t||A - PLTL^tP^t||\tdx/x[' + str(dtype) + ']\tdb/b[' + str(dtype) + ']\n')
    res_2.write('RAND DIM:\tCOND:\t||A - PLDL^tP^t||\tdx/x[' + str(dtype) + ']\tdb/b[' + str(dtype) + ']\n')
    for i in xrange(min_size, max_size + 1, step):
        print 'step:', i
        test_solution = np.ones(i, dtype=dtype)
        h = np.random.randn(i, i)
        if positive_def:
            h = dot(h.T, h)
        h = (h + h.T) / 2
        cond = np.linalg.cond(h)
        h = np.array(h, dtype=f128)

        free_values = dot(h, test_solution)

        P, L, cell_sizes, tridiagonal = bunch_kaufman(h)
        diags = [1,0,-1]
        T = np.array(sparse.spdiags(tridiagonal, diags, i, i, format='csc').todense(), dtype=np.float128)
        assembled_result = dot(dot(dot(dot(P, L), T), np.array(np.matrix(L).getH())), P.T)
        factor_difference_1 = frobenius_norm(h - assembled_result)

        x = symmetric_system_solve(h, free_values)
        rel_err_1 = relative_error(test_solution, x)

        computed_free_variables = dot(h, x)
        residual_1 = relative_error(free_values, computed_free_variables)

        LD, P = cholesky_diagonal(h)
        D = np.zeros([i, i], dtype=f128)
        diag = LD.diagonal()
        np.fill_diagonal(D, diag)
        L = tril(LD, -1) + I(i)
        assembled_result = dot(dot(L, D), L.T)
        for (idx1, idx2) in reversed(P):
            exchange_rows(assembled_result, idx1, idx2)

        for (idx1, idx2) in reversed(P):
            exchange_columns(assembled_result, idx1, idx2)
        factor_difference_2 = frobenius_norm(h - assembled_result)

        x = linear_cholesky_solve(LD, P, free_values)
        rel_err_2 = relative_error(test_solution, x)

        computed_free_variables = dot(h, x)
        residual_2 = relative_error(free_values, computed_free_variables)

        res_1.write(str(i) + '\t')
        res_1.write(str(cond) + '\t')
        res_1.write(str(factor_difference_1) + '\t')
        res_1.write(str(rel_err_1) + '\t')
        res_1.write(str(residual_1) + '\n')
        res_2.write(str(i) + '\t')
        res_2.write(str(cond) + '\t')
        res_2.write(str(factor_difference_2) + '\t')
        res_2.write(str(rel_err_2) + '\t')
        res_2.write(str(residual_2) + '\n')
    res_1.close()
    res_2.close()



def extended_ckhol_linear_solve_random_test(max_size=50, step=1, dtype=np.float64, avg=10, positive_def=False):
    dtype=np.array([1], dtype=dtype).dtype
    if max_size < 11:
        max_size = 11
    if positive_def:
        res = open(str(max_size) + "_linear_random_positive_chol_" + str(dtype) + ".txt", "w")
    else:
        res = open(str(max_size) + "_linear_random_chol_" + str(dtype) + ".txt", "w")
    res.write('RAND DIM:\tCOND AVG:\tCOND MAX:\t||A - PLDL^tP^t|| AVG\t||A - PLDL^tP^t|| MAX\tdx/x[' + str(dtype) + ']AVG\tdx/x[' + str(dtype) + ']MAX\tdb/b[' + str(dtype) + ']AVG\tdb/b[' + str(dtype) + ']MAX\n')
    for i in xrange(10, max_size + 1, step):
        cond_avg = 0
        cond_max = 0
        factor_difference_avg = 0
        factor_difference_max = 0
        relative_error_avg = 0
        relative_error_max = 0
        residual_avg = 0
        residual_max = 0
        for j in xrange(avg):
            test_solution = np.ones(i, dtype=dtype)
            h = np.random.randn(i, i)
            if positive_def:
                h = dot(h.T, h)
            h = (h + h.T) / 2
            cc = np.linalg.cond(h)
            h = np.array(h, dtype=f128)
            cond_avg += cc
            if cc > cond_max: cond_max = cc

            free_values = dot(h, test_solution)

            LD, P = cholesky_diagonal(h)
            D = np.zeros([i, i], dtype=f128)
            diag = LD.diagonal()
            np.fill_diagonal(D, diag)
            L = tril(LD, -1) + I(i)
            assembled_result = dot(dot(L, D), L.T)
            for (idx1, idx2) in reversed(P):
                exchange_rows(assembled_result, idx1, idx2)

            for (idx1, idx2) in reversed(P):
                exchange_columns(assembled_result, idx1, idx2)
            cdiff = frobenius_norm(h - assembled_result)
            factor_difference_avg += cdiff
            if cdiff > factor_difference_max: factor_difference_max = cdiff

            x = linear_cholesky_solve(LD, P, free_values)
            rel_err = relative_error(test_solution, x)
            relative_error_avg += rel_err
            if rel_err > relative_error_max: relative_error_max = rel_err

            computed_free_variables = dot(h, x)
            cresidual = relative_error(free_values, computed_free_variables)
            residual_avg += cresidual
            if cresidual > residual_max: residual_max = cresidual
        cond_avg /= avg
        factor_difference_avg /= avg
        relative_error_avg /= avg
        residual_avg /= avg

        res.write(str(i))
        res.write('\t')
        res.write(str(cond_avg) + '\t' + str(cond_max) + '\t')
        res.write(str(factor_difference_avg) + '\t' + str(factor_difference_max) + '\t')
        res.write(str(relative_error_avg) + '\t' + str(relative_error_avg) + '\t')
        res.write(str(residual_avg) + '\t' + str(residual_avg) + '\n')


def extended_double_linear_random_test(max_size=50, step=1, dtype=np.float64, avg=10, positive_def=False):
    dtype=np.array([1], dtype=dtype).dtype
    if max_size < 11:
        max_size = 11
    if positive_def:
        res_1 = open(str(max_size) + "_linear_random_positive_bunch_" + str(dtype) + ".txt", "w")
        res_2 = open(str(max_size) + "_linear_random_positive_chol_" + str(dtype) + ".txt", "w")
    else:
        res_1 = open(str(max_size) + "_linear_random_bunch_" + str(dtype) + ".txt", "w")
        res_2 = open(str(max_size) + "_linear_random_chol_" + str(dtype) + ".txt", "w")
    res_1.write('RAND DIM:\tCOND AVG:\tCOND MAX:\t||A - PLTL^tP^t|| AVG\t||A - PLTL^tP^t|| MAX\tdx/x[' + str(dtype) + ']AVG\tdx/x[' + str(dtype) + ']MAX\tdb/b[' + str(dtype) + ']AVG\tdb/b[' + str(dtype) + ']MAX\n')
    res_2.write('RAND DIM:\tCOND AVG:\tCOND MAX:\t||A - PLDL^tP^t|| AVG\t||A - PLDL^tP^t|| MAX\tdx/x[' + str(dtype) + ']AVG\tdx/x[' + str(dtype) + ']MAX\tdb/b[' + str(dtype) + ']AVG\tdb/b[' + str(dtype) + ']MAX\n')
    for i in xrange(10, max_size + 1, step):
        print 'step:', i
        cond_avg_1 = 0
        cond_max_1 = 0
        factor_difference_avg_1 = 0
        factor_difference_max_1 = 0
        relative_error_avg_1 = 0
        relative_error_max_1 = 0
        residual_avg_1 = 0
        residual_max_1 = 0
        cond_avg_2 = 0
        cond_max_2 = 0
        factor_difference_avg_2 = 0
        factor_difference_max_2 = 0
        relative_error_avg_2 = 0
        relative_error_max_2 = 0
        residual_avg_2 = 0
        residual_max_2 = 0
        for j in xrange(avg):
            test_solution = np.ones(i, dtype=dtype)
            h = np.random.randn(i, i)
            if positive_def:
                h = dot(h.T, h)
            h = (h + h.T) / 2
            cc = np.linalg.cond(h)
            h = np.array(h, dtype=f128)
            cond_avg_1 += cc
            if cc > cond_max_1: cond_max_1 = cc
            cond_avg_2 += cc
            if cc > cond_max_2: cond_max_2 = cc

            free_values = dot(h, test_solution)

            P, L, cell_sizes, tridiagonal = bunch_kaufman(h)
            diags = [1,0,-1]
            T = np.array(sparse.spdiags(tridiagonal, diags, i, i, format='csc').todense(), dtype=np.float128)
            assembled_result = dot(dot(dot(dot(P, L), T), np.array(np.matrix(L).getH())), P.T)
            cdiff = frobenius_norm(h - assembled_result)
            factor_difference_avg_1 += cdiff
            if cdiff > factor_difference_max_1: factor_difference_max_1 = cdiff

            LD, P = cholesky_diagonal(h)
            D = np.zeros([i, i], dtype=f128)
            diag = LD.diagonal()
            np.fill_diagonal(D, diag)
            L = tril(LD, -1) + I(i)
            assembled_result = dot(dot(L, D), L.T)
            for (idx1, idx2) in reversed(P):
                exchange_rows(assembled_result, idx1, idx2)

            for (idx1, idx2) in reversed(P):
                exchange_columns(assembled_result, idx1, idx2)
            cdiff = frobenius_norm(h - assembled_result)
            factor_difference_avg_2 += cdiff
            if cdiff > factor_difference_max_2: factor_difference_max_2 = cdiff

            x = symmetric_system_solve(h, free_values)
            rel_err = relative_error(test_solution, x)
            relative_error_avg_1 += rel_err
            if rel_err > relative_error_max_1: relative_error_max_1 = rel_err

            x = linear_cholesky_solve(LD, P, free_values)
            rel_err = relative_error(test_solution, x)
            relative_error_avg_2 += rel_err
            if rel_err > relative_error_max_2: relative_error_max_2 = rel_err

            computed_free_variables = dot(h, x)
            cresidual = relative_error(free_values, computed_free_variables)
            residual_avg_1 += cresidual
            if cresidual > residual_max_1: residual_max_1 = cresidual

            computed_free_variables = dot(h, x)
            cresidual = relative_error(free_values, computed_free_variables)
            residual_avg_2 += cresidual
            if cresidual > residual_max_2: residual_max_2 = cresidual
        cond_avg_1 /= avg
        factor_difference_avg_1 /= avg
        relative_error_avg_1 /= avg
        residual_avg_1 /= avg
        cond_avg_2 /= avg
        factor_difference_avg_2 /= avg
        relative_error_avg_2 /= avg
        residual_avg_2 /= avg

        res_1.write(str(i))
        res_1.write('\t')
        res_1.write(str(cond_avg_1) + '\t' + str(cond_max_1) + '\t')
        res_1.write(str(factor_difference_avg_1) + '\t' + str(factor_difference_max_1) + '\t')
        res_1.write(str(relative_error_avg_1) + '\t' + str(relative_error_avg_1) + '\t')
        res_1.write(str(residual_avg_1) + '\t' + str(residual_avg_1) + '\n')
        res_2.write(str(i))
        res_2.write('\t')
        res_2.write(str(cond_avg_2) + '\t' + str(cond_max_2) + '\t')
        res_2.write(str(factor_difference_avg_2) + '\t' + str(factor_difference_max_2) + '\t')
        res_2.write(str(relative_error_avg_2) + '\t' + str(relative_error_avg_2) + '\t')
        res_2.write(str(residual_avg_2) + '\t' + str(residual_avg_2) + '\n')



def factorization_test(mtx):
    """Tests Bunch-Kaufman factorization for a given matrix.

        Test by factorizing it, restoring from factors and counting difference.

    Args:
         (np.array): testing matrix

    Returns:
        none

    Raises:
        Exception: An error occurred when Bunch-Kaufman doesn't work properly.

    """
    P, L, cell_sizes, tridiagonal = bunch_kaufman(mtx.copy())
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
    print 'Tridiagonal'
    print tridiagonal
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
    if precondition:
        check_values = dot(mtx_precision, computed_result / diag_r) / diag_l
    else:
        check_values = dot(mtx_precision, computed_result)
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
    dtype=mtx.dtype
    original_solve = np.zeros(mtx.shape[0]) + 1
    original_solve_precision = np.zeros(mtx.shape[0], dtype=np.dtype) + 1
    if original_solve.shape[0] != mtx.shape[0]:
        raise Exception('Sizes of matrix and original solve must be equal!')
    mtx_precision = np.array(mtx, dtype=np.dtype)
    free_variables = mtx.dot(original_solve)
    free_variables_precision = mtx_precision.dot(original_solve_precision)

    kaufmann_result = symmetric_system_solve(mtx, np.array(free_variables_precision, dtype=np.dtype))

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


def time_test(min_size=10, max_size=60, avg=10):
    import timeit
    import os
    os.chdir('time_tests')
    res = open(str(min_size) + ':' + str(max_size) + ':time', 'w')
    res.write('n\tb\tc\n')
    for i in xrange(min_size, max_size + 1, 1):
        print 'step', i
        h = hilb(i, dtype=f128)
        res.write(str(i) + '\t')
        h_copies_b = [h.copy() for _ in xrange(avg)]
        h_copies_c = [h.copy() for _ in xrange(avg)]
        res.write(str(timeit.timeit(lambda: bunch_kaufman(h_copies_b.pop()), number=avg) / avg) + '\t')
        res.write(str(timeit.timeit(lambda: cholesky_diagonal(h_copies_c.pop()), number=avg) / avg) + '\n')
    res.close()
    os.chdir('..')