import numpy as np
from math import factorial

# hilbert matrix (Hij = 1/(i + j - 1))

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