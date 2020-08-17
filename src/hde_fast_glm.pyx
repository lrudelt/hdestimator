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


def counts_C(np.ndarray[np.double_t, ndim=1] spiketimes, double t_bin, double T_0, double T_f, str mode):
    cdef int N_bins = int((T_f - T_0) / t_bin)
    cdef np.ndarray[np.double_t, ndim = 1] sptimes = np.sort(np.append(spiketimes, [(N_bins + 1) * t_bin]))
    cdef np.ndarray[np.int_t, ndim = 1] counts = np.zeros(N_bins, dtype=int)
    cdef double t_up, t_low
    cdef int j_low, j_up
    t_up = t_bin
    j_low = 0
    j_up = 0
    t_low = 0.
    for i in range(N_bins):
        while sptimes[j_low] < t_low:
            j_low += 1
        while sptimes[j_up] < t_up:
            j_up += 1
        t_up += t_bin
        t_low += t_bin
        counts[i] += j_up - j_low
    if mode == 'binary':
        counts[np.nonzero(counts)] = 1
    return counts


def past_activity(np.ndarray[np.double_t, ndim=1] spiketimes, np.ndarray[np.int_t, ndim=1] indices, int d_past, int N_trials, double kappa, double tau, double t_bin, double T_0, double T_f, str mode):
    cdef int N_bins = int((T_f - T_0) / t_bin)
    cdef int N_spikes = len(spiketimes)
    cdef np.ndarray[np.double_t, ndim= 1] sptimes = np.sort(np.append(spiketimes, [N_bins * t_bin]))
    cdef np.ndarray[DTYPE_t, ndim= 1] past = np.zeros([N_trials * d_past], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim= 1] medians = np.zeros(d_past, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim= 1] past_temp = np.zeros(N_bins, dtype=DTYPE)
    cdef double t_up, t_low, t_low_mem, s
    cdef int j, j_low, j_up, m
    cdef double T = T_f - T_0
    t_up = 0.
    if mode == 'medians':
        t_up = 0.
        for k in range(d_past):
            j_low = 0
            j_up = 0
            t_low = t_up - tau * np.power(10, k * kappa)
            t_low_mem = t_low
            for i in range(N_bins):
                while sptimes[j_low] < t_low:
                    j_low += 1
                while sptimes[j_up] < t_up:
                    j_up += 1
                t_up += t_bin
                t_low += t_bin
                past_temp[i] = j_up - j_low
            t_up = t_low_mem
            medians[k] = np.sort(past_temp)[int(N_bins / 2.)] + 1
        t_up = 0.
        for k in range(d_past):
            m = medians[k]
            j = 0
            index = indices[0]
            j_low = 0
            j_up = 0
            t_low = t_up - tau * np.power(10, k * kappa)
            t_low_mem = t_low
            for i in range(N_bins):
                while sptimes[j_low] < t_low:
                    j_low += 1
                while sptimes[j_up] < t_up:
                    j_up += 1
                t_up += t_bin
                t_low += t_bin
                if i == index:
                    if j_up - j_low >= m:
                        past[j + k * N_trials] = 1
                    j += 1
                    index = indices[j]
            t_up = t_low_mem
    if mode == 'general':
        for k in range(d_past):
            j = 0
            index = indices[0]
            j_low = 0
            j_up = 0
            t_low = t_up - tau * np.power(10, k * kappa)
            t_low_mem = t_low
            for i in range(N_bins):
                while sptimes[j_low] < t_low:
                    j_low += 1
                while sptimes[j_up] < t_up:
                    j_up += 1
                t_up += t_bin
                t_low += t_bin
                if i == index:
                    if j_up - j_low > 0:
                        past[j + k * N_trials] = j_up - j_low
                    j += 1
                    index = indices[j]
            t_up = t_low_mem
    if mode == 'binary':
        for k in range(d_past):
            j = 0
            index = indices[0]
            j_low = 0
            j_up = 0
            t_low = t_up - tau * np.power(10, k * kappa)
            t_low_mem = t_low
            for i in range(N_bins):
                while sptimes[j_low] < t_low:
                    j_low += 1
                while sptimes[j_up] < t_up:
                    j_up += 1
                t_up += t_bin
                t_low += t_bin
                if i == index:
                    if j_up - j_low > 0:
                        past[j + k * N_trials] = 1
                    j += 1
                    index = indices[j]
            t_up = t_low_mem
    return past


def lograte_sum(np.ndarray[DTYPE_t, ndim=1] past, np.ndarray[np.double_t, ndim=1] kernel, int d_past, int N_trials):
    cdef np.ndarray[np.double_t, ndim= 1] lograte = np.zeros(N_trials, dtype=np.double)
    for k in range(d_past):
        lograte += kernel[k] * past[k * N_trials:(k + 1) * N_trials]
    return lograte


def jac_sum(np.ndarray[DTYPE_t, ndim=1] past, np.ndarray[np.int_t, ndim=1] counts, np.ndarray[np.double_t, ndim=1] reciproke_rate, int d_past, int N_trials):
    cdef np.ndarray[np.double_t, ndim= 1] jac = np.zeros(d_past, dtype=np.double)
    for k in range(d_past):
        jac[k] = np.dot(past[k * N_trials:(k + 1) * N_trials], counts) - \
            np.dot(past[k * N_trials:(k + 1) * N_trials], reciproke_rate)
    return jac


def L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d_past, int N_trials, np.ndarray[np.double_t, ndim=1] kernel, double mu):
    cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d_past, N_trials) + mu
    cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
    cdef double L = np.dot(counts, log_rate) - np.sum(np.log(1 + rate))
    return L


def jac_L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d_past, int N_trials, np.ndarray[np.double_t, ndim=1] kernel, double mu):
    cdef int n_sp = np.sum(counts)
    cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d_past, N_trials) + mu
    cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
    cdef np.ndarray[np.double_t, ndim = 1] reciproke_rate = np.multiply(np.power(1 + rate, -1), rate)
    cdef np.ndarray[np.double_t, ndim = 1] jac_kernel = jac_sum(past, counts, reciproke_rate, d_past, N_trials)
    cdef double dmu = n_sp - np.sum(reciproke_rate)
    return np.append([dmu], jac_kernel)


def hess_L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d_past, int N_trials, np.ndarray[np.double_t, ndim=1] kernel, double mu):
    cdef int dtot = d_past + 1
    cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d_past, N_trials) + mu
    cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
    cdef np.ndarray[np.double_t, ndim = 1] reciproke_rate = np.multiply(np.power(1 + rate, -2), rate)
    cdef np.ndarray[np.double_t, ndim = 2] hess = np.diag(np.zeros(dtot))
    # Compute elements involving mu
    hess[0][0] = -np.sum(reciproke_rate)
    for l in np.arange(1, dtot):
        hess[0][l] = hess[l][0] = - \
            np.dot(past[(l - 1) * N_trials:l * N_trials], reciproke_rate)
    # Compute all other elements
    for j in np.arange(1, dtot):
        for l in np.arange(j, dtot):
            hess[j][l] = hess[l][j] = -np.dot(past[(l - 1) * N_trials:l * N_trials], np.multiply(
                past[(j - 1) * N_trials:j * N_trials], reciproke_rate))
    return hess


def H_cond_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d_past, int N_trials, np.ndarray[np.double_t, ndim=1] kernel, double mu):
    return -L_B_past(counts, past, d_past, N_trials, kernel, mu) / N_trials
