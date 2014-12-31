import numpy as np
from numpy import identity as I, dot
from kaufmann_solver.utils.utils import transposition_matrix, separate_permutation, triangular_inversion
from operator import itemgetter


def inv_1_2(mtx):
    if mtx.shape == (1, 1):
        if mtx[0, 0] == 0:
            raise Exception('Matrix of size 1x1 is singular')
        return np.array([[1.0/mtx[0, 0]]])
    if mtx.shape != (2, 2):
        raise Exception("Matrix isn't 2x2 matrix")
    a = mtx[0, 0]
    b = mtx[0, 1]
    c = mtx[1, 0]
    d = mtx[1, 1]
    det = a*d - b*c
    if det == 0:
        raise Exception('Matrix of size 2x2 is singular')
    inverse = np.zeros(mtx.shape)
    inverse[0, 0] = d/det
    inverse[0, 1] = -c/det
    inverse[1, 0] = -b/det
    inverse[1, 1] = a/det
    return inverse


def bunch_kaufmann(mtx_origin, alpha):
    if len(mtx_origin.shape) != 2:
        raise Exception("Matrix isn't 2D")
    if mtx_origin.shape[0] != mtx_origin.shape[1]:
        raise Exception('Square matrix expected, matrix of shape ' + str(mtx_origin.shape) + ' is given')
    if not np.array_equal(mtx_origin.T, mtx_origin):
        raise Exception('Symmetric matrix expected')
    mtx = mtx_origin.copy()
    
    n = mtx.shape[0]
    sum = 0
    cell_sizes = []
    L, P = I(n), I(n)
    while sum < n:
        mtxs = mtx[sum: n, sum: n]
        idx = max([(abs(mtxs[j][j]), j) for j in xrange(mtxs.shape[0])], key=itemgetter(0))[1]

        permutation_step = transposition_matrix(n, sum, sum + idx)
        permutation = permutation_step[sum: n, sum: n]
        triangular_step = I(n)
        triangular = triangular_step[sum: n, sum: n]
        # conjugate M' with permutation matrix
        mtxs[:,:] = dot(dot(permutation, mtxs), permutation.T)[:,:]

        #P = permutation_step.dot(P).dot(np.matrix(permutation_step).getI())
        # find index for larger column abs and this abs
        [lambda_val, idx] = max([(abs(mtxs[j][0]), j) for j in xrange(mtxs.shape[0])], key=itemgetter(0))
        if abs(mtxs[0][0]) >= alpha*lambda_val:
            n_k = 1
            if mtx.shape[0] <= sum + n_k:
                cell_sizes.append(n_k)
                break
            permutation[:,:] = I(mtxs.shape[0])[:,:]
        else:
            [sigma_val, j_idx] = max([(abs(mtxs[j][idx]), j) for j in xrange(mtxs.shape[0]) if j != idx], key=itemgetter(0))
            if sigma_val*abs(mtxs[0][0]) >= alpha*lambda_val**2:
                n_k = 1
                if mtx.shape[0] <= sum + n_k:
                    cell_sizes.append(n_k)
                    break
                permutation[:,:] = I(mtxs.shape[0])[:,:]
            else:
                if abs(mtxs[idx][idx]) >= alpha*sigma_val:
                    n_k = 1
                    if mtx.shape[0] <= sum + n_k:
                        cell_sizes.append(n_k)
                        break
                    permutation[:,:] = transposition_matrix(mtxs.shape[0], 0, idx)[:,:]

                else:
                    n_k = 2
                    if mtx.shape[0] <= sum + n_k:
                        cell_sizes.append(n_k)
                        break
                    permutation[:,:] = dot(transposition_matrix(mtxs.shape[0], 2, idx), transposition_matrix(mtxs.shape[0], 1, j_idx))[:,:]
        mtxs_image = np.dot(np.dot(permutation, mtxs), permutation)
        T_k = mtxs_image[0:n_k, 0:n_k]
        #T_k_inverse = np.matrix(T_k.copy()).getI()
        T_k_inverse = inv_1_2(T_k)
        B_k = mtxs_image[n_k: mtxs_image.shape[0], 0: n_k]

        triangular[n_k:triangular.shape[0], 0:n_k] = -B_k.dot(T_k_inverse)

        #mtxs[:,:] = triangular.dot(permutation).dot(mtxs).dot(permutation.T).dot(np.matrix(triangular).getH())[:,:]
        mtxs[:,:] = dot(dot(dot(dot(triangular, permutation), mtxs), permutation.T), np.matrix(triangular).getH())[:,:]
        print '-'*80
        print "M':"
        print mtxs
        print '-'*80

        """print 'Get inversion of:'
        print triangular_step
        print 'Inversion is'
        inv = -triangular_step + 2*np.identity(triangular_step.shape[0])
        print inv
        print 'Check(T * T^-1):'
        print triangular_step.dot(inv)
        print 'And (T^-1 * T):'
        print inv.dot(triangular_step)"""

        print '-'*80, '\n', '-'*80
        print 'TRIANGULAR_STEP'
        print triangular_step
        print '-'*80, '\n', '-'*80
        P = P.dot(permutation_step.T).dot(triangular_inversion(triangular_step))
        print P
        sum += n_k
        cell_sizes.append(n_k)
    print '-'*80
    print P.dot(mtx).dot(np.matrix(P).getH())
    return mtx, P, cell_sizes


def linear_system_solve(system_matrix_origin, free_values, alpha):
    system_matrix = system_matrix_origin.copy()
    tridiagonal, PL, cell_sizes = bunch_kaufmann(system_matrix, alpha)
    P, L = separate_permutation(PL)
    print '-'*80
    print 'Tridiagonal'
    print tridiagonal
    print '-'*80
    print 'Permutation'
    print P
    print '-'*80
    print 'Low-triangular'
    print L
    print '-'*80
    print 'Matrix:'
    print system_matrix
    print '-'*80
    print 'with bunch kaufmann:'
    print P.dot(L).dot(tridiagonal).dot(np.matrix(L).getH()).dot(P.T)
    print '-'*80


    """
    We need to solve some systems:
    Lz = P^t b,
    Tw = z,
    L^* y = w,
    x = Py
    """

    z = np.linalg.solve(L, P.T.dot(free_values))
    w = np.linalg.solve(tridiagonal, z)
    y = np.linalg.solve(np.matrix(L).getH(), w)
    return P.dot(y)
