# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, srand
cdef extern from "stdlib.h":
        int RAND_MAX
        int INT_MAX
from libc.math cimport log


def sumlog(np.ndarray[np.double_t, ndim=1] data):
    # speed up calculation of sum of log by calculating log of product
    cdef double res = 0.0
    cdef double tmp = 1.0
    for i in range(data.shape[0]):
        tmp *= data[i]
        if not (1e-14 < tmp < 1e14):
            res += log(tmp)
            tmp = 1.0
    res += log(tmp)
    return res

def zrecursionNstate(np.ndarray[np.double_t, ndim=3] A,
                     double alpha, double beta,
                     int niter=1000, seed=None):
    """
    Recursion for switching betwen N states in two environments 
    
    seed should be in [2, uint32max]
    """
    cdef double dRAND_MAX = <double> RAND_MAX
    if seed is None:
        seed = np.random.randint(2, INT_MAX)
    srand(seed)
    # set x to 1 with probability pienv
    cdef unsigned int N = A.shape[1]
    cdef int x = (alpha/(alpha+beta)) < (rand() / dRAND_MAX)
    cdef np.ndarray[np.double_t, ndim=1] n = np.ones(N, dtype = np.double) / N
    cdef np.ndarray[np.double_t, ndim=1] nnew = np.zeros(N, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1] zs = np.zeros(niter, dtype = np.double)
    cdef double z, prob
    cdef unsigned int c, i, j
    for c in range(niter):
        prob = rand() / dRAND_MAX
        if x:
            x = prob > beta
        else:
            x = prob < alpha
        for i in range(N):
            nnew[i] = 0.0
        for i in range(N):
            for j in range(N):
                nnew[i] += A[x, i, j] * n[j] 
        z = 0.0
        for i in range(N):
            z += nnew[i]
        for i in range(N):
            n[i] = nnew[i]/z
        zs[c] = z
    return zs
