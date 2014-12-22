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
        n_k = 0
        if abs(mtx_shiffer[0][0]) >= alpha*lambda_val:
            n_k = 1
            if mtx.shape[0] <= sum + n_k:
                break
            # make identity permutation matrix
            permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, 0)[:,:]
        else:
            # same as above, for idx-th column of mtx_shiffer (without diagonal element)
            [sigma_val, j_idx] = max([(abs(mtx_shiffer[j][idx]), j) for j in xrange(mtx_shiffer.shape[0]) if j != idx], key=itemgetter(0))
            if sigma_val*abs(mtx_shiffer[0][0]) >= alpha*lambda_val**2:
                n_k = 1
                if mtx.shape[0] <= sum + n_k:
                    break
                # make identity permutation matrix
                permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, 0)[:,:]
            else:
                if abs(mtx_shiffer[idx][idx]) >= alpha*sigma_val:
                    n_k = 1
                    if mtx.shape[0] <= sum + n_k:
                        break
                    # make non-identity permutation matrix (0, idx) for mtx_shiffer
                    permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 0, idx)[:,:]

                else:
                    n_k = 2
                    if mtx.shape[0] <= sum + n_k:
                        break
                    permutation[:,:] = transposition_matrix(mtx_shiffer.shape[0], 2, idx).dot(transposition_matrix(mtx_shiffer.shape[0], 1, j_idx))[:,:]
        # conjugate M' with permutation matrix
        mtx_shiffer_image = np.dot(np.dot(permutation, mtx_shiffer), permutation)
        T_k = mtx_shiffer_image[0:n_k, 0:n_k]
        T_k_inverse = np.matrix(T_k.copy()).getI()
        B_k = mtx_shiffer_image[n_k: mtx_shiffer_image.shape[0], 0: n_k]

        # Fill L_k
        triangular[n_k:triangular.shape[0], 0:n_k] = -B_k.dot(np.array(T_k_inverse))

        #print '\n\n\n'
        #print 'triangular step:'
        #print triangular_step
        #print '-'*80
        #print 'permutation step:'
        #print permutation_step

        #print sum, ':' , permutation_step.shape, ':', triangular_step.shape, ':', P.shape

        #print '-'*80
        #print 'after left remove:'
        #print triangular.dot(permutation).dot(mtx_shiffer)

        #print 'after right remove:'
        #print mtx_shiffer.dot(permutation.T).dot(np.matrix(triangular).getH())

        #print "mtx' after left & right remove:"
        #print triangular.dot(permutation).dot(mtx_shiffer).dot(permutation.T).dot(np.matrix(triangular).getH())
        #print '-'*80

        mtx_shiffer[:,:] = triangular.dot(permutation).dot(mtx_shiffer).dot(permutation.T).dot(np.matrix(triangular).getH())[:,:]

        #print 'mtx after left & right remove:'
        #print mtx
        #print '-'*80

        #mtx_shiffer = triangular.dot(permutation.dot(mtx_shiffer).dot(permutation.T).dot(np.matrix(triangular).getH()))

        P = triangular_step.dot(permutation_step).dot(P)

        #print '-'*80
        #print 'Inverse of permutation with L:'
        #print np.matrix(P).getI()
        #print '-'*80

        #print '-'*80
        #print 'Inverse of hermit transpose of P with L:'
        #print np.matrix(P).getH().getI()
        #print '-'*80

        #print '-'*80
        #print 'Check:'
        #print np.matrix(P).getI().dot(mtx).dot(np.matrix(P).getH().getI())
        #print '-'*80

        #print '-'*80
        #print 'Result matrix:'
        #print mtx
        #print '-'*80

        sum += n_k
        cell_sizes.append(n_k)

    #print '-'*80
    #print 'Result matrix:'
    #print mtx
    #print '-'*80
    return mtx, P, cell_sizes