__author__ = 'Const'
import numpy as np
from numpy import identity as I, dot, float128 as f128, argmax, outer
from kaufmann_solver.utils.bunch_kaufman_utils import exchange_rows, exchange_columns, partial_left_one, partial_right_one


def cholesky_diagonal(mtx_origin):
    mtx = np.array(mtx_origin, dtype=f128)
    n = mtx.shape[0]
    P = []
    for k in xrange(n):
        mtx_view = mtx[k:, k:]
        diag = mtx_view.diagonal()
        j = argmax(diag)
        mtx[[k, k + j], :] = mtx[[k + j, k], :]
        mtx[:, [k, k + j]] = mtx[:, [k + j, k]]
        P.append((k, k + j))
        alpha = mtx_view[0, 0]
        v = mtx_view[1:, 0].copy()
        mtx_view[1:, 0] = v / alpha
        mtx_view[1:, 1:] -= outer(v, v) / alpha
    return mtx, P

