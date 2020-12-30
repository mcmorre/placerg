#dice6.pyx
import numpy as np
cimport numpy as np
cimport cython # so we can use cython decorators

from libc.stdlib cimport rand, RAND_MAX

@cython.wraparound(False)
@cython.boundscheck(False)
def dice6(np.ndarray[np.float64_t, ndim=3]  P):
    cdef int l = P.shape[0] 
    cdef int m = P.shape[1]
    cdef int n = P.shape[2]
    cdef int i
    cdef int j
    cdef int k
    cdef int q
    cdef np.ndarray samples = np.zeros((l, m, n)) 
    cdef list choices = [0,1]
    cdef int r = len(choices)

    for i in range(l):
        for j in range(m):
            for k in range(n):
                if float(rand())/float(RAND_MAX) > P[i,j,k]:
                    samples[i,j,k] += 1
    return samples



