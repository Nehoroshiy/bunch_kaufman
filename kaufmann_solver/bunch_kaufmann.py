import numpy as np
from kaufmann_solver.utils.utils import transposition_matrix
from operator import itemgetter

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
                break
            permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, 0)[:,:]
        else:
            [sigma_val, j_idx] = max([(abs(mtx_shiffer[j][idx]), j) for j in xrange(mtx_shiffer.shape[0]) if j != idx], key=itemgetter(0))
            if sigma_val*abs(mtx_shiffer[0][0]) >= alpha*lambda_val**2:
                n_k = 1
                if mtx.shape[0] <= sum + n_k:
                    break
                permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, 0)[:,:]
            else:
                if abs(mtx_shiffer[idx][idx]) >= alpha*sigma_val:
                    n_k = 1
                    if mtx.shape[0] <= sum + n_k:
                        break
                    permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, idx)[:,:]

                else:
                    n_k = 2
                    if mtx.shape[0] <= sum + n_k:
                        break
                    permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 2, idx).dot(transposition_matrix(mtx_shiffer.shape[0], 1, j_idx))[:,:]
        mtx_shiffer_image = np.dot(np.dot(permutation, mtx_shiffer), permutation)
        T_k = mtx_shiffer_image[0:n_k, 0:n_k]
        T_k_inverse = np.matrix(T_k.copy()).getI()
        B_k = mtx_shiffer_image[n_k: mtx_shiffer_image.shape[0], 0: n_k]

        triangular[n_k:triangular.shape[0], 0:n_k] = -B_k.dot(np.array(T_k_inverse))

        mtx_shiffer[:,:] = triangular.dot(permutation).dot(mtx_shiffer).dot(permutation.T).dot(np.matrix(triangular).getH())[:,:]

        P = triangular_step.dot(permutation_step).dot(P)
        sum += n_k
        cell_sizes.append(n_k)
    return mtx, P, cell_sizes