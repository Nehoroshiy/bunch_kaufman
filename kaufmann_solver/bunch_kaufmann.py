import numpy as np
from kaufmann_solver.utils.utils import transposition_matrix, separate_permutation
from operator import itemgetter

def inv_1_2(mtx):
    if mtx.shape == (1, 1):
        assert mtx[0, 0] != 0, "matrix is null-determinant"
        return np.array([[1.0/mtx[0, 0]]])
    assert mtx.shape == (2, 2), "matrix isn't 2x2 matrix"
    a = mtx[0, 0]
    b = mtx[0, 1]
    c = mtx[1, 0]
    d = mtx[1, 1]
    det = a*d - b*c
    assert det != 0, 'matrix is null-determinant'
    print 'det:', det

    inverse = np.zeros(mtx.shape)
    inverse[0, 0] = d/det
    inverse[0, 1] = -c/det
    inverse[1, 0] = -b/det
    inverse[1, 1] = a/det
    return inverse

def bunch_kaufmann(mtx, alpha):
    # make identity L matrix
    L = transposition_matrix(mtx.shape[0], 0, 0)
    # make identity global permutation matrix
    P = transposition_matrix(mtx.shape[0], 0, 0)
    sum = 0
    cell_sizes = []
    while sum < mtx.shape[0]:
        # i-th step of algorithm: we work with n[i]-th mtx_shiffer
        mtx_shiffer = mtx[sum: mtx.shape[0], sum: mtx.shape[1]]
        # find index with largest diagonal abs
        idx = max([(abs(mtx_shiffer[j][j]), j) for j in xrange(mtx_shiffer.shape[0])], key=itemgetter(0))[1]
        # make permutation matrix for (0, idx) transposition
        permutation_step = transposition_matrix(mtx.shape[0], sum, sum + idx)
        permutation = permutation_step[sum: mtx.shape[0], sum: mtx.shape[1]]
        # make triangular matrix
        triangular_step = transposition_matrix(mtx.shape[0], 0, 0)
        triangular = triangular_step[sum: mtx.shape[0], sum: mtx.shape[1]]
        # conjugate M' with permutation matrix
        mtx_shiffer[:,:] = np.array(permutation.dot(np.matrix(mtx_shiffer)).dot(np.matrix(permutation).getI()))

        #P = permutation_step.dot(P).dot(np.matrix(permutation_step).getI())
        # find index for larger column abs and this abs
        [lambda_val, idx] = max([(abs(mtx_shiffer[j][0]), j) for j in xrange(mtx_shiffer.shape[0])], key=itemgetter(0))
        if abs(mtx_shiffer[0][0]) >= alpha*lambda_val:
            n_k = 1
            if mtx.shape[0] <= sum + n_k:
                cell_sizes.append(n_k)
                break
            permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, 0)[:,:]
        else:
            [sigma_val, j_idx] = max([(abs(mtx_shiffer[j][idx]), j) for j in xrange(mtx_shiffer.shape[0]) if j != idx], key=itemgetter(0))
            if sigma_val*abs(mtx_shiffer[0][0]) >= alpha*lambda_val**2:
                n_k = 1
                if mtx.shape[0] <= sum + n_k:
                    cell_sizes.append(n_k)
                    break
                permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, 0)[:,:]
            else:
                if abs(mtx_shiffer[idx][idx]) >= alpha*sigma_val:
                    n_k = 1
                    if mtx.shape[0] <= sum + n_k:
                        cell_sizes.append(n_k)
                        break
                    permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, idx)[:,:]

                else:
                    n_k = 2
                    if mtx.shape[0] <= sum + n_k:
                        cell_sizes.append(n_k)
                        break
                    permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 2, idx).dot(transposition_matrix(mtx_shiffer.shape[0], 1, j_idx))[:,:]
        mtx_shiffer_image = np.dot(np.dot(permutation, mtx_shiffer), permutation)
        T_k = mtx_shiffer_image[0:n_k, 0:n_k]
        #T_k_inverse = np.matrix(T_k.copy()).getI()
        T_k_inverse = inv_1_2(T_k)
        B_k = mtx_shiffer_image[n_k: mtx_shiffer_image.shape[0], 0: n_k]

        triangular[n_k:triangular.shape[0], 0:n_k] = -B_k.dot(T_k_inverse)

        mtx_shiffer[:,:] = triangular.dot(permutation).dot(mtx_shiffer).dot(permutation.T).dot(np.matrix(triangular).getH())[:,:]
        print '-'*80
        print "M':"
        print mtx_shiffer
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

        P = P.dot(permutation_step.T).dot(-triangular_step + 2*np.identity(triangular_step.shape[0]))
        print P
        sum += n_k
        cell_sizes.append(n_k)
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

    # solve first system to find z
    z = np.linalg.solve(L, P.T.dot(free_values))
    print 'z:', z
    # solve second system to find w
    w = np.linalg.solve(tridiagonal, z)
    print 'w:', w
    # solve third system to find y
    y = np.linalg.solve(np.matrix(L).getH(), w)
    print 'y:', y
    # return the result (P*y)
    return P.dot(y)
