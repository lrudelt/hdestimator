# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:30:11 2018

@author: lucas
"""

"""Functions"""
import seaborn.apionly as sns
from scipy.optimize import bisect
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import matplotlib
import numpy as np
import sys
sys.path.append('../../Functions')
#import plotutils

load_path = "../../../../Data/Simulated/analysis_Data/"

"""Parameters"""
duration = '90min'
p_dict = {'1min': 0.01, '5min': 0.01, '10min': 0.02,
          '20min': 0.02, '45min': 0.02, '90min': 0.02}
p = p_dict[duration]
T_0 = 100. - 10**(-4)
# shift time to avoid spiketimes on bin edges, 100 s burnin
T_f = 54100.
T = T_f - T_0  # 90 min recording
t_bin = 0.005
N_d = 30
d_max = 50
d_min = 1
kappa_d = bisect(lambda kappa: np.sum(
    [d_min * np.power(10, i * kappa) for i in range(N_d)]) - d_max, 0, 1.)
d = 0
d_list = []
for i in range(N_d):
    d += np.power(10, kappa_d * i)
    d_list += [d]

d_list = np.array(d_list).astype(int)
kappa = 0.0  # linear binning
mode = 'medians'
binning_mode = 'equalbins'
tau = 0.02
N_kappa = 25
Tm_0 = 0.01
Tm_f = 3.  # Simulated
N_Tm = 50
Tm_list = np.zeros(N_Tm)
kappa_Tm = bisect(lambda kappa: np.sum(
    Tm_0 * np.power(10, np.arange(50) * kappa)) - Tm_f, -1., 10)
Tm = 0
for k in range(N_Tm):
    Tm += Tm_0 * np.power(10, k * kappa_Tm)
    Tm_list[k] = Tm


"""Load data """

"""Simulated"""
M_max = np.loadtxt("%sM_max_5ms.dat" % load_path)
if duration == '90min':
    M_BBC = np.loadtxt('%sM_NSB_exp_opt_90min_p%d.dat' % (load_path, p * 100))
    M_GLM_NSB_exp = np.loadtxt(
        '%sM_GLM_exp_opt_90min_p%d.dat' % (load_path, p * 100))
    M_BBC_mean = np.loadtxt('%sM_NSB_opt_mean_%s_p%d.dat'
                            % (load_path, duration, int(p * 100)))
    M_BBC_std = np.loadtxt('%sM_NSB_opt_std_%s_p%d.dat'
                           % (load_path, duration, int(p * 100)))
else:
    M_BBC = np.loadtxt('%sM_NSB_exp_opt_iid_%s_k%d_dmax%d_p%d.dat' %
                       (load_path, duration, N_kappa, d_max, p * 100))  # result based on 90 min iid drawn samples
    M_BBC_mean = np.loadtxt('%sM_NSB_opt_iid_mean_%s_k%d_dmax%d_p%d.dat'
                            % (load_path, duration, N_kappa, d_max, int(p * 100)))
    M_BBC_std = np.loadtxt('%sM_NSB_opt_iid_std_%s_k%d_dmax%d_p%d.dat'
                           % (load_path, duration, N_kappa, d_max, int(p * 100)))

M_shuffled = np.loadtxt('%sM_sh_plugin_exp_opt_iid_%s_k%d_dmax%d.dat' %
                        (load_path, duration, N_kappa, d_max))
M_shuffled_mean = np.loadtxt('%sM_sh_plugin_opt_iid_mean_%s_k%d_dmax%d.dat'
                             % (load_path, duration, N_kappa, d_max))
M_shuffled_std = np.loadtxt('%sM_sh_plugin_opt_iid_std_%s_k%d_dmax%d.dat'
                            % (load_path, duration, N_kappa, d_max))

# M_GLM_NSB_exp=np.loadtxt('%sM_GLM_NSB_exp_opt_90min_k%d_dmax%d_p%d.dat'%(N_kappa,d_max,p*100))
M_GLM_BIC = np.delete(np.loadtxt('%sM_GLM_max.dat' % load_path), (23, 34, 42))

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '14.0'
matplotlib.rcParams['xtick.labelsize'] = '14'
matplotlib.rcParams['ytick.labelsize'] = '14'
matplotlib.rcParams['legend.fontsize'] = '14'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ((ax)) = plt.subplots(1, 1, figsize=(3.5, 2.8))

#fig.set_size_inches(4, 3)

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
soft_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
# sns.palplot(sns.color_palette("RdBu_r", 15))  #visualize the color palette
# ax.set_color_cycle(sns.color_palette("coolwarm_r",num_lines)) setting it
# as color cycle to do automatised color assignment

##########################################
########## Simulated Conventional ########
##########################################


ax.set_xscale('log')
ax.set_xlim((0.01, 3.))
#ax.set_xticks(np.array([1, 10, 50]))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.spines['bottom'].set_bounds(0.01, 3)
ax.set_xlabel(r'past range $T$ [sec]')

##### y-axis ####
ax.set_ylabel(r'history dependence $R(T)$')
ax.set_ylim((0.0, .16))
ax.set_yticks([0.0, 0.05, 0.10, 0.15])
ax.spines['left'].set_bounds(.0, 0.15)


##### Unset Borders #####
ax.spines['top'].set_bounds(0, 0)
ax.spines['right'].set_bounds(0, 0)

# a11.text(np.log(15), 0.17, r'$\hat{M}$',
#         color='0.05', ha='left', va='bottom')

# ax.text(0.04, 0.136, r'$M_{max}$',
#         color='0.5', ha='left', va='bottom')


#ax1.plot([np.log(50)], [0.1], marker=1, linewidth=1.0, color='firebrick')


#


# removing the axis ticks
#


# a11.text(np.log(15), 0.17, r'$\hat{M}$',
#         color='0.05', ha='left', va='bottom')

# ax1.text(6, 0.136, r'$M_{max}$',
#         color='0.5', ha='left', va='bottom')
#
# ax.text(.02, 0.134, r'$R^*_{model}$',
#         color='0.35', ha='left', va='bottom')
ax.plot([Tm_0, Tm_f], [M_max, M_max], '--', color='0.5', zorder=1)
# ax.plot(Tm_list[np.argmax(M_BBC)], [np.amax(M_GLM_NSB_exp)], color=0.5,
# alpha=0.95,zorder=3)#, label='Model'
ax.plot(np.delete(Tm_list, (23, 34, 42)), M_GLM_BIC, linewidth=1.2,
        color='.5', label='true', zorder=8)
# ax.plot(Tm_list[np.argmax(M_shuffled])], [np.amax(M_shuffled)], color=0.5, alpha=0.95,zorder=3)#, label='Model'
#ax.plot(Tm_list, M_GLM_NSB_exp, color=soft_red, alpha=0.7,zorder=3)
ax.plot(Tm_list, M_BBC, linewidth=1.2,  color=main_red,
        label=r'BBC', zorder=4)
ax.fill_between(Tm_list, M_BBC - M_BBC_std, M_BBC +
                M_BBC_std, facecolor=main_red, alpha=0.3)
ax.plot(Tm_list, M_shuffled, linewidth=1.2,
        color=main_blue, label=r'Shuffling', zorder=3)
ax.fill_between(Tm_list, M_shuffled - M_shuffled_std,
                M_shuffled + M_shuffled_std, facecolor=main_blue, alpha=0.3)
#ax1.errorbar(d_list, M_PT_Simulated, yerr=M_PT_Simulated_err, color='C1', label='PT')
ax.legend(loc=(.38, .02), frameon=False)

#ax1.plot([np.log(50)], [0.1], marker=1, linewidth=1.0, color='firebrick')
#ax1.legend(loc=2, fontsize=10., frameon=False)


fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# plt.savefig('../Mopt_vs_Tm_model_%s.pdf' % duration,
#             format="pdf", bbox_inches='tight')
# plt.savefig('Poster/Mopt_vs_Tm_model_%s.png' % duration,
#             format="png", dpi=600, bbox_inches='tight')
plt.savefig('../../Paper_Figures/fig2_benchmarks/Ropt_vs_T_model_%s.pdf' % duration,
            format="pdf", bbox_inches='tight')
# plt.savefig('../Mopt_vs_Tm_model_%s.png' % duration,
#             format="png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()
