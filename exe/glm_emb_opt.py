from cython_functions_thinkpad2 import*
import os
from scipy.io import loadmat
from scipy.optimize import bisect
from scipy.optimize import minimize
from scipy import spatial
from scipy.special import polygamma
from scipy.integrate import quad
from cython_functions import*
import pickle
import random
import Gnuplot
import numpy as np
import sys
sys.path.append('../Functions')
#import statsmodels.api as sm
execfile('../Functions/nsb_estimator.py')
execfile('../Functions/AIS_discr.py')
execfile('../Functions/AIS_data.py')
execfile('../Functions/AIS_GLM.py')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""MAIN"""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""Indices for computation on the cluster"""
Tm_index = (int(os.environ['SGE_TASK_ID']) - 1) % 50
d_index = (int(os.environ['SGE_TASK_ID']) - 1) / 50

"""Parameters"""
#############################################################
########### Parameters for the simulated recordings #########
#############################################################
# shift time to avoid spiketimes on bin edges, 100 s burnin
T_0 = 100. - 10**(-4)
T_f = 54100.  # 10 times 90 min
T = T_f - T_0  # 90 min recording
t_bin = 0.005
mode = 'general'
###########################################################
### parameters for stepwise upsampling of GLM estimator ###
###########################################################
downsampling = 20
upsampling = 2.5
tolerance = 0.0002
###########################################################
##### parameters for opt d and kappa and fixed Tm   #######
###########################################################
d_list = [10, 20, 40, 60, 80, 100, 120, 150]
d_emb = d_list[d_index]
tau_emb = 0.0005
Tm_0 = 0.01
Tm_f = 3.
N_Tm = 50
Tm_list = np.zeros(N_Tm)
kappa_Tm = bisect(lambda kappa: np.sum(
    Tm_0 * np.power(10, np.arange(50) * kappa)) - Tm_f, -1., 10)
Tm = 0
for k in range(N_Tm):
    Tm += Tm_0 * np.power(10, k * kappa_Tm)
    Tm_list[k] = Tm
Tm_emb = Tm_list[Tm_index]
print Tm_emb
kappa_emb = bisect(lambda kappa: np.sum(
    tau_emb * np.power(10, np.arange(d_emb) * kappa)) - Tm_emb, -1., 10)

"""Preprocess data"""
spiketimes = np.loadtxt('../../Data/Simulated/spiketimes_constI_5ms.dat')
spiketimes = spiketimes - T_0
counts = counts_C(spiketimes, t_bin, T_0, T_f, 'binary')
N_bins = len(counts)

"""Spike entropy"""
P_spike = np.sum(counts) / float(N_bins)
H_spike = -P_spike * np.log(P_spike) - (1 - P_spike) * np.log(1 - P_spike)

""" Compute M and BIC """
H_cond, BIC = _H_cond_BIC_GLM(spiketimes, counts, N_bins, d_emb, kappa_emb,
                              tau_emb, t_bin, T_0, T_f, mode, downsampling, upsampling, tolerance)
M_GLM = [(H_spike - H_cond) / H_spike]
np.savetxt('../../Data/Simulated/M_GLM_range%d_d%d.dat' % (Tm_index, d), M_GLM)
np.savetxt('../../Data/Simulated/BIC_range%d_d%d.dat' % (Tm_index, d), [BIC])


"""Experimental analysis"""

###########################################################
##################### Functions  #######################
###########################################################
sys.path.append('../Functions')
# import statsmodels.api as sm
execfile('../Functions/nsb_estimator.py')
execfile('../Functions/AIS_discr.py')
execfile('../Functions/AIS_data.py')
execfile('../Functions/AIS_GLM.py')

###########################################################
##### Indices for computation on the cluster   #######
###########################################################
# neuron_index = (int(os.environ['SGE_TASK_ID']) - 1)
neuron_index = 0
# setting = sys.argv[1]
setting = 'Culture'
params_out = True

# TODO: Compute M_GLM vs T_m for same opt embeddings as best performing model free

if True:
    ###########################################################
    ##################### Parameters  #######################
    ###########################################################
    execfile('params_GLM.py')

    ###########################################################
    ##################### Load data  #######################
    ###########################################################
    execfile('load_data_GLM_%s.py' % setting)

    ###########################################################
    #####################   Main   #######################
    ###########################################################

    """Preprocess data"""
    spiketimes_self = spiketimes_self - T_0
    # spiketimes_PPA = spiketimes_PPA - T_0
    counts = counts_C(spiketimes_self, t_bin, T_0, T_f, 'binary')
    N_bins = len(counts)

    """Spike entropy"""
    P_spike = np.sum(counts) / float(N_bins)
    H_spike = -P_spike * np.log(P_spike) - (1 - P_spike) * np.log(1 - P_spike)

    # H_cond_PPA=np.loadtxt('../../Data/In_vivo/analysis_Data/GLM/H_cond_PPA_opt.dat')[1]
    # (H_spike-H_cond_PPA)/H_spike
    # M_GLM = np.loadtxt('../../Data/In_vivo/analysis_Data/GLM/M_GLM_neuron%d.dat'% neuron)
    # M_GLM_cond_PPA = np.loadtxt(
    #     '../../Data/In_vivo/analysis_Data/GLM/M_GLM_cond_PPA_neuron%d.dat'% neuron)
    # M_GLM - M_GLM_cond_PPA
    """ Compute M and BIC """
    M_GLM = []
    # M_GLM_cond_PPA = []
    for i, Tm in enumerate(Tm_list[5:6]):
        d_self = int(d_opt_list[i])
        kappa_self = kappa_opt_list[i]
        kappa_self = 0
        tau_self = Tm / np.sum([10**(kappa_self * k)
                                for k in np.arange(d_self)])
        H_cond_self, BIC, mu_self, h_self = _H_cond_GLM_single(
            spiketimes_self, d_self, kappa_self, tau_self, mode_self, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out)
        print d_self, kappa_self, tau_self
        # H_cond_combined, BIC = _H_cond_GLM_combined(spiketimes_self, d_self, kappa_self, tau_self, mode_self,
        # spiketimes_PPA, d_PPA, kappa_PPA, tau_PPA,  mode_PPA, counts, N_bins, t_bin, T_0, T_f,  downsampling_ratio, mu_self, h_self)
        M_GLM += [(H_spike - H_cond_self) / H_spike]
        # M_GLM_cond_PPA += [(H_cond_PPA - H_cond_combined) / H_spike]
    # np.savetxt('%s/GLM/M_GLM_neuron%d.dat' %               (ana_data_path, neuron), M_GLM)
    # np.savetxt('%s/GLM/M_GLM_cond_PPA_neuron%d.dat' %               (ana_data_path, neuron), M_GLM_cond_PPA)


# Compute M_GLM_self and M_GLM_PPA for optimal embedding as it was
# found using BIC
if False:
    M_GLM = []
    M_GLM_cond_PPA = []
    for neuron_index in range(28):
        ###########################################################
        ##################### Parameters  #######################
        ###########################################################
        execfile('params_GLM.py')

        ###########################################################
        ##################### Load data  #######################
        ###########################################################
        execfile('load_data_GLM_%s.py' % setting)

        ###########################################################
        #####################   Main   #######################
        ###########################################################

        """Preprocess data"""
        spiketimes_self = spiketimes_self - T_0
        spiketimes_PPA = spiketimes_PPA - T_0
        counts = counts_C(spiketimes_self, t_bin, T_0, T_f, 'binary')
        N_bins = len(counts)

        """Spike entropy"""
        P_spike = np.sum(counts) / float(N_bins)
        H_spike = -P_spike * np.log(P_spike) - \
            (1 - P_spike) * np.log(1 - P_spike)

        H_cond_self, BIC, mu_self, h_self = _H_cond_GLM_single(
            spiketimes_self, d_self, kappa_self, tau_self, mode_self, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out)
        H_cond_combined, BIC = _H_cond_GLM_combined(spiketimes_self, d_self, kappa_self, tau_self, mode_self,
                                                    spiketimes_PPA, d_PPA, kappa_PPA, tau_PPA,  mode_PPA, counts, N_bins, t_bin, T_0, T_f,  downsampling_ratio, mu_self, h_self)
        print H_cond_combined, H_cond_self
        M_GLM += [(H_spike - H_cond_self) / H_spike]
        M_GLM_cond_PPA += [(H_cond_PPA - H_cond_combined) / H_spike]
    np.savetxt('%s/M_GLM_self_opt.dat' %
               (ana_data_path), M_GLM)
    np.savetxt('%s/M_GLM_cond_PPA_opt.dat' %
               (ana_data_path), M_GLM_cond_PPA)
