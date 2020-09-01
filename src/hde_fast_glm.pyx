from __future__ import division
import numpy as np
import random
from scipy.special import factorial
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint8
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint8_t DTYPE_t

# Counts of spieks in the past bin of current spiking


def counts_C(np.ndarray[np.double_t, ndim=1] spiketimes, double t_bin, double T_0, double T_f, str embedding_mode):
    cdef int N = int((T_f - T_0) / t_bin)
    cdef np.ndarray[np.double_t, ndim = 1] sptimes = np.sort(np.append(spiketimes, [(N + 1) * t_bin]))
    cdef np.ndarray[np.int_t, ndim = 1] counts = np.zeros(N, dtype=int)
    cdef double t_up, t_low
    cdef int j_low, j_up
    t_up = t_bin
    j_low = 0
    j_up = 0
    t_low = 0.
    for i in range(N):
        while sptimes[j_low] < t_low:
            j_low += 1
        while sptimes[j_up] < t_up:
            j_up += 1
        t_up += t_bin
        t_low += t_bin
        counts[i] += j_up - j_low
    if embedding_mode == 'binary':
        counts[np.nonzero(counts)] = 1
    return counts

# Embedding of past activity for the product with a discrete past kernel to compute the firing rate of the GLM


def past_activity(np.ndarray[np.double_t, ndim=1] spiketimes, int d, double kappa, double tau, double t_bin, int N, str embedding_mode):
    cdef np.ndarray[np.double_t, ndim= 1] sptimes = np.sort(np.append(spiketimes, [N * t_bin]))
    cdef np.ndarray[DTYPE_t, ndim= 1] past = np.zeros([N * d], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim= 1] medians = np.zeros(d, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim= 1] past_temp = np.zeros(N, dtype=DTYPE)
    cdef double t_up, t_low, t_low_mem, s
    cdef int j, j_low, j_up, m
    t_up = 0.
    if embedding_mode == 'medians':
        t_up = 0.
        for k in range(d):
            j_low = 0
            j_up = 0
            t_low = t_up - tau * np.power(10, k * kappa)
            t_low_mem = t_low
            for i in range(N):
                while sptimes[j_low] < t_low:
                    j_low += 1
                while sptimes[j_up] < t_up:
                    j_up += 1
                t_up += t_bin
                t_low += t_bin
                past_temp[i] = j_up - j_low
            t_up = t_low_mem
            medians[k] = np.sort(past_temp)[int(N / 2.)] + 1
        t_up = 0.
        for k in range(d):
            m = medians[k]
            j_low = 0
            j_up = 0
            t_low = t_up - tau * np.power(10, k * kappa)
            t_low_mem = t_low
            for i in range(N):
                while sptimes[j_low] < t_low:
                    j_low += 1
                while sptimes[j_up] < t_up:
                    j_up += 1
                t_up += t_bin
                t_low += t_bin
                if j_up - j_low >= m:
                    past[i + k * N] = 1
            t_up = t_low_mem
    if embedding_mode == 'counts':
        for k in range(d):
            j_low = 0
            j_up = 0
            t_low = t_up - tau * np.power(10, k * kappa)
            t_low_mem = t_low
            for i in range(N):
                while sptimes[j_low] < t_low:
                    j_low += 1
                while sptimes[j_up] < t_up:
                    j_up += 1
                t_up += t_bin
                t_low += t_bin
                if j_up - j_low > 0:
                    past[i + k * N] = j_up - j_low
            t_up = t_low_mem
    return past


def downsample_past_activity(np.ndarray[DTYPE_t, ndim=1] past, np.ndarray[np.int_t, ndim=1] indices, int N, int d):
    cdef int N_downsampled = len(indices)
    cdef np.ndarray[DTYPE_t, ndim= 1] past_downsampled = np.zeros([N_downsampled * d], dtype=DTYPE)
    for k in range(d):
        for i, index in enumerate(indices):
            past_downsampled[i + k *
                             N_downsampled] = past[index + k * N]
    return past_downsampled


def lograte_sum(np.ndarray[DTYPE_t, ndim=1] past, np.ndarray[np.double_t, ndim=1] kernel, int d, int N):
    cdef np.ndarray[np.double_t, ndim= 1] lograte = np.zeros(N, dtype=np.double)
    for k in range(d):
        lograte += kernel[k] * past[k * N:(k + 1) * N]
    return lograte


def jac_sum(np.ndarray[DTYPE_t, ndim=1] past, np.ndarray[np.int_t, ndim=1] counts, np.ndarray[np.double_t, ndim=1] reciproke_rate, int d, int N):
    cdef np.ndarray[np.double_t, ndim= 1] jac = np.zeros(d, dtype=np.double)
    for k in range(d):
        jac[k] = np.dot(past[k * N:(k + 1) * N], counts) - \
            np.dot(past[k * N:(k + 1) * N], reciproke_rate)
    return jac

# Bernoulli likelihood of the spiketrain for given past and GLM parameters h and mu


def L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d, int N, np.ndarray[np.double_t, ndim=1] kernel, double mu):
    cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d, N) + mu
    cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
    cdef double L = np.dot(counts, log_rate) - np.sum(np.log(1 + rate))
    return L

# Jacobian of the Bernoulli likelihood of the spiketrain for given past and GLM parameters h and mu


def jac_L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d, int N, np.ndarray[np.double_t, ndim=1] kernel, double mu):
    cdef int n_sp = np.sum(counts)
    cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d, N) + mu
    cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
    cdef np.ndarray[np.double_t, ndim = 1] reciproke_rate = np.multiply(np.power(1 + rate, -1), rate)
    cdef np.ndarray[np.double_t, ndim = 1] jac_kernel = jac_sum(past, counts, reciproke_rate, d, N)
    cdef double dmu = n_sp - np.sum(reciproke_rate)
    return np.append([dmu], jac_kernel)

# Hessian of the Bernoulli likelihood of the spiketrain for given past and GLM parameters h and mu


def hess_L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d, int N, np.ndarray[np.double_t, ndim=1] kernel, double mu):
    cdef int dtot = d + 1
    cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d, N) + mu
    cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
    cdef np.ndarray[np.double_t, ndim = 1] reciproke_rate = np.multiply(np.power(1 + rate, -2), rate)
    cdef np.ndarray[np.double_t, ndim = 2] hess = np.diag(np.zeros(dtot))
    # Compute elements involving mu
    hess[0][0] = -np.sum(reciproke_rate)
    for l in np.arange(1, dtot):
        hess[0][l] = hess[l][0] = - \
            np.dot(past[(l - 1) * N:l * N], reciproke_rate)
    # Compute all other elements
    for j in np.arange(1, dtot):
        for l in np.arange(j, dtot):
            hess[j][l] = hess[l][j] = -np.dot(past[(l - 1) * N:l * N], np.multiply(
                past[(j - 1) * N:j * N], reciproke_rate))
    return hess

# Estimate of the conditional entropy based on an average of the likelihood over the data set


def H_cond_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d, int N, double mu, np.ndarray[np.double_t, ndim=1] kernel):
    return -L_B_past(counts, past, d, N, kernel, mu) / N
