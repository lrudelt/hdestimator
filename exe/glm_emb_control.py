import argparse
from sys import exit, stderr, argv, path
from os.path import isfile, isdir, realpath, dirname, exists
import h5py
import ast
import yaml
import numpy as np
from _version import __version__

ESTIMATOR_DIR = dirname(realpath(__file__))
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))

if 'hde_fast_glm' not in sys.modules:
    from hde_fast_glm as import*

__version__ = "unknown"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
#
#
# from scipy.io import loadmat
# from scipy.optimize import bisect
# from scipy.optimize import minimize
# from scipy import spatial
# from scipy.special import polygamma
# from scipy.integrate import quad
# import os
# import pickle
# import random
# import numpy as np
# import sys
# sys.path.append('../Functions')
# if 'cython_functions_cluster' not in sys.modules:
#     from cython_functions_thinkpad2 import*
# # import Gnuplot
# # import statsmodels.api as sm
# # from AIS_KDE import*
# # execfile('AIS_KDE.py')
# execfile('../Functions/nsb_estimator.py')
# # execfile('AIS_NN.py')
# execfile('../Functions/AIS_discr.py')
# execfile('../Functions/AIS_data.py')
# execfile('../Functions/AIS_GLM.py')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""MAIN"""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""Indices for computation on the cluster"""
###########################################################
##### parameters for opt d and kappa and fixed Tp   #######
###########################################################
# Tp_index = (int(os.environ['SGE_TASK_ID']) - 1)
# Tp_index=5
###########################################################
######## parameters for true M estimated by GLM  ##########
###########################################################
# Tp_index = (int(os.environ['SGE_TASK_ID']) - 1) % 50
# d_index = (int(os.environ['SGE_TASK_ID']) - 1) / 50 % 8
# tau_index = (int(os.environ['SGE_TASK_ID']) - 1) / 400 + 1

"""Parameters"""
#############################################################
########### Parameters for the simulated recordings #########
#############################################################
# shift time to avoid spiketimes on bin edges, 100 s burnin
T_0 = 100. - 10**(-4)
T_f = 54100.  # 10 times 90 min
T = T_f - T_0  # 90 min recording
t_bin = 0.005
###########################################################
### parameters for stepwise upsampling of GLM estimator ###
###########################################################
downsampling = 20
downsampling_ratio = 0.05
upsampling = 2.5
tolerance = 0.001  # this did not work so well, better
tolerance = 0.0002  # Or maybe only one downsampled and then compute for full sample

"""Preprocess data"""
spiketimes = np.loadtxt(
    '../../Data/Simulated/simulated_Data/spiketimes_constI_5ms.dat')
spiketimes = spiketimes - T_0
counts = counts_C(spiketimes, t_bin, T_0, T_f, 'binary')
N_bins = len(counts)

"""Spike entropy"""
P_spike = np.sum(counts) / float(N_bins)
H_spike = -P_spike * np.log(P_spike) - (1 - P_spike) * np.log(1 - P_spike)

""" Compute GLM for max opt params """
sampling = sys.argv[1]
rec_length = sys.argv[2]
number_samples = 10
Tp_index = (int(os.environ['SGE_TASK_ID']) - 1) / number_samples
sample_index = (int(os.environ['SGE_TASK_ID']) - 1) % number_samples
# sample_index = 0
# Tp_index = 0
# Tp_index = 0  # dummy variable
# setting = 'Retina'

R_GLM_NSB_H = []
R_GLM_NSB_R = []
R_GLM_sh_plugin = []
execfile('params_embedding_opt.py')
# for Tp_index in range(N_Tp):
for Tp_index in np.arange(Tp_index, Tp_index + 1):
    # M_NSB_exp=np.loadtxt('../../Data/Simulated/M_NSB_exp_opt_%s_p%d.dat' %(rec_length, p * 100))
    kappa_NSB_H = np.load('../../Data/Simulated/analysis_Data/kappa_exp_opt_NSB_%s_%s_k%d_dmax%d_p%d_H.npy' %
                          (sampling + str(sample_index), rec_length,  N_kappa, dmax_emb_opt, p * 100))[Tp_index]
    kappa_NSB_R = np.load('../../Data/Simulated/analysis_Data/kappa_exp_opt_NSB_%s_%s_k%d_dmax%d_p%d_R.npy' %
                          (sampling + str(sample_index), rec_length,  N_kappa, dmax_emb_opt, p_R * 100))[Tp_index]
    kappa_sh_plugin = np.load('../../Data/Simulated/analysis_Data/kappa_exp_opt_sh_plugin_%s_%s_k%d_dmax%d.npy' %
                              (sampling + str(sample_index), rec_length,  N_kappa, dmax_emb_opt))[Tp_index]
    d_NSB_H = np.load('../../Data/Simulated/analysis_Data/d_exp_opt_NSB_%s_%s_k%d_dmax%d_p%d_H.npy' %
                      (sampling + str(sample_index), rec_length,  N_kappa, dmax_emb_opt, p * 100)).astype(int)[Tp_index]
    d_NSB_R = np.load('../../Data/Simulated/analysis_Data/d_exp_opt_NSB_%s_%s_k%d_dmax%d_p%d_R.npy' %
                      (sampling + str(sample_index), rec_length,  N_kappa, dmax_emb_opt, p_R * 100)).astype(int)[Tp_index]
    d_sh_plugin = np.load('../../Data/Simulated/analysis_Data/d_exp_opt_sh_plugin_%s_%s_k%d_dmax%d.npy' %
                          (sampling + str(sample_index), rec_length,  N_kappa, dmax_emb_opt)).astype(int)[Tp_index]
    Tp = Tp_list[Tp_index]
    tau_NSB_H = Tp / np.sum([10**(kappa_NSB_H * k)
                             for k in np.arange(d_NSB_H)])
    tau_NSB_R = Tp / np.sum([10**(kappa_NSB_R * k)
                             for k in np.arange(d_NSB_R)])
    tau_sh_plugin = Tp / \
        np.sum([10**(kappa_sh_plugin * k) for k in np.arange(d_sh_plugin)])

    """Compute GLM estimates on full recording"""
    H_cond_NSB_H, BIC_H, mu_NSB_H, h_NSB_H = _H_cond_GLM_single(
        spiketimes, d_NSB_H, kappa_NSB_H, tau_NSB_H, mode, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out=True)
    # H_cond_NSB_R, BIC_R, mu_NSB_R, h_NSB_R = _H_cond_GLM_single(
    #     spiketimes, d_NSB_R, kappa_NSB_R, tau_NSB_R, mode, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out=True)
    H_cond_sh_plugin, BIC, mu_sh_plugin, h_sh_plugin = _H_cond_GLM_single(
        spiketimes, d_sh_plugin, kappa_sh_plugin, tau_sh_plugin, mode, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out=True)

    R_GLM_NSB_H += [(H_spike - H_cond_NSB_H) / H_spike]
    # R_GLM_NSB_R += [(H_spike - H_cond_NSB_H) / H_spike]
    R_GLM_sh_plugin += [(H_spike - H_cond_sh_plugin) / H_spike]

    """Compute GLM estimates on downsampled recording (this probably doesn't make so much sense)"""

    # T_f = T_0 + T_rec
    # N_sample = int(T_rec / t_bin)
    # indices_inference = np.arange(N_sample).astype(int)
    # counts_inference = counts[indices_inference]
    # indices_inference = np.append(indices_inference, 0)
    # # NSB
    # y_t_NSB = past_activity(
    #     spiketimes, indices_inference, d_NSB, N_sample, kappa_NSB, tau_NSB, t_bin, T_0, T_f, mode)
    # H_cond_NSB_downsampled = H_cond_B_past(
    #     counts_inference, y_t_NSB, d_NSB, N_sample, h_NSB, mu_NSB)
    # # Shuffling
    # y_t_sh_plugin = past_activity(
    #     spiketimes, indices_inference, d_sh_plugin, N_sample, kappa_sh_plugin, tau_sh_plugin, t_bin, T_0, T_f, mode)
    # H_cond_sh_plugin_downsampled = H_cond_B_past(
    #     counts_inference, y_t_sh_plugin, d_sh_plugin, N_sample, h_sh_plugin, mu_sh_plugin)
    #
    # M_GLM_NSB_downsampled = [(H_spike - H_cond_NSB_downsampled) / H_spike]
    # M_GLM_sh_plugin_downsampled = [
    #     (H_spike - H_cond_sh_plugin_downsampled) / H_spike]

    """Save results"""

    np.save('%s/Rtot_GLM_BBC_exp_%s_%s_dmax%d_p%d_H_range%d.npy' %
            (data_path, sampling + str(sample_index), rec_length, dmax_emb_opt, p * 100, Tp_index), R_GLM_NSB_H)
    # np.save('../../Data/Simulated/analysis_Data/Rtot_GLM_BBC_exp_%s_%s_dmax%d_p%d_R.npy' %
    #         (sampling, rec_length, dmax_emb_opt, p_R * 100), R_GLM_NSB_R)
    np.save('%s/Rtot_GLM_sh_plugin_exp_%s_%s_dmax%d_range%d.npy' %
            (data_path, sampling + str(sample_index), rec_length, dmax_emb_opt, Tp_index), R_GLM_sh_plugin)
# np.savetxt('../../Data/Simulated/analysis_Data/Mtot_GLM_BBC_exp_%s_%s_dmax%d_p%d_H_downsampled.dat' %
#            (sampling, rec_length, dmax_emb_opt, p * 100), M_GLM_NSB_downsampled)
# np.savetxt(
#     '../../Data/Simulated/analysis_Data/Mtot_GLM_sh_plugin_exp_%s_%s_dmax%d_downsampled.dat' % (sampling, rec_length, dmax_emb_opt), M_GLM_sh_plugin_downsampled)
