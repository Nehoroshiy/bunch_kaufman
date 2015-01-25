from numpy import dot
from kaufmann_solver.utils.utils import euclid_vector_norm
from math import sqrt


def conjugate_gradients(A, b, x0, tolerance=1e-16):
    k = 0
    residual = b - dot(A, x0)
    computed_result = x0
    q = 0
    q_prev = 0
    c = 0
    norm = euclid_vector_norm(residual)
    while norm > tolerance:
        q_prev = q
        q = residual / norm
        k += 1
        a_step = dot(dot(q, A), q)
        if k == 1:
            d = a_step
            v = norm / d
            c = q
        else:
            l = norm / d
            d = a_step - norm * l
            v = - norm * v / d
            c = q - l * c
        computed_result += v * c
        residual = dot(A, q) - a_step * q - norm * q_prev
        norm = euclid_vector_norm(residual)
    return computed_result


def conjugate_gradients_pract(A, b, x0, tolerance=1e-17):
    k = 0
    x = x0.copy()
    r = b - dot(A, x)
    ro_c = dot(r, r)
    delta = tolerance * euclid_vector_norm(b)
    p = 0
    ro_m = 0
    while sqrt(ro_c) >= delta:
        k += 1
        if k == 1:
            p = r
        else:
            tau = ro_c / ro_m
            p = r + tau * p
        w = dot(A, p)
        mu = ro_c / dot(p, w)
        x += mu * p
        r -= mu * w
        ro_m = ro_c
        ro_c = dot(r, r)
        if k > 50000:
            break

    return x, k


def preconditioned_conjugate_gradients_diag(A, b, x0, tolerance=1e-17):
    M = A.diagonal()
    k = 0
    r = b - dot(A, x0)
    x = x0.copy()
    z = M * r
    z_prev = 0
    r_prev = 0
    while euclid_vector_norm(r) > tolerance:
        k += 1
        if k == 1:
            p = z
        else:
            tau = dot(r, z) / dot(r_prev, z_prev)
            p = z + tau * p
        mu = dot(r, z) / dot(dot(p, A), p)
        x -= mu * p
        r_prev = r
        r -= mu * dot(A, p)
        z_prev = z
        z = M * r
    return x, k