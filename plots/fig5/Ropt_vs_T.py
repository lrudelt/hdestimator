"""Functions"""
import seaborn.apionly as sns
from scipy.optimize import bisect
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import matplotlib
import numpy as np
from scipy.io import loadmat
import sys
sys.path.append('../../Scripts/Functions')
# import plotutils

"""Params"""

recording = 'Neuropixel'
recorded_system = 'V1'
sampling = 'consecutive'
rec_length = '40min'
neuron_index = 24
dmax_emb_opt = 5
execfile('../../../../Scripts/Paper/params_embedding_opt.py')
# EC: 12, rostrolaterArea: 1, 56, 24

"""Load data"""
if recording == 'Neuropixel':
    execfile('../../../../Scripts/Paper/load_data_Neuropixel.py')
    neuron_str = '{}-{}'.format(neuron[0], neuron[1])
if recording == 'Other':
    execfile('../../../../Scripts/Paper/load_data_{}.py'.format(recorded_system))
    neuron_str = str(neuron)

dmax_emb_opt = 5
neuron_str = '2-338'
# '2-303': 104 freak
# '2-338' : 52 normal
# '2-357' : 62 bursty

R_BBC = np.load('%sR_NSB_exp_opt_%s_%s_k%d_dmax%d_p%d_neuron-%s.npy' %
                (plottingDataDir, sampling, rec_length,  N_kappa, dmax_emb_opt, p * 100, neuron_str))
R_shuffled = np.load('%sR_sh_plugin_exp_opt_%s_%s_k%d_dmax%d_neuron-%s.npy' %
                     (plottingDataDir, sampling, rec_length,  N_kappa, dmax_emb_opt, neuron_str))
R_BBC_samples = np.load('%sR_NSB_exp_opt_samples_%s_%s_k%d_dmax%d_p%d_neuron-%s.npy' %
                        (plottingDataDir, sampling_bootstraps, rec_length,  N_kappa, dmax_emb_opt, p * 100, neuron_str), allow_pickle=True)
R_shuffled_samples = np.load('%sR_sh_plugin_exp_opt_samples_%s_%s_k%d_dmax%d_neuron-%s.npy' %
                             (plottingDataDir, sampling_bootstraps, rec_length,  N_kappa, dmax_emb_opt, neuron_str), allow_pickle=True)
R_BBC_std = np.std(R_BBC_samples, axis=1).astype(float)
R_shuffled_std = np.std(R_shuffled_samples, axis=1).astype(float)
# R_BBC_samples_sorted = np.sort(R_BBC_samples, axis=1).astype(float)
# R_shuffled_samples_sorted = np.sort(
#     R_shuffled_samples, axis=1).astype(float)

# R_BBC_fivebins = np.load('%sR_NSB_exp_opt_%s_%s_k%d_dmax%d_p%d_neuron-%d-%d.npy' %
#                          (analysisDataDir, sampling, rec_length,  N_kappa, 5, p * 100, neuron[0], neuron[1]))

# R_BBC_iQDist = R_BBC_samples_sorted[np.argmax(
#     R_BBC)][14] - R_BBC_samples_sorted[np.argmax(
#         R_BBC)][4]
# R_shuffled_iQDist = R_shuffled_samples_sorted[np.argmax(
#     R_shuffled)][14] - R_shuffled_samples_sorted[np.argmax(
#         R_shuffled)][4]
# R_BBC_90 = R_BBC_samples_sorted[:, 17]
# R_BBC_10 = R_BBC_samples_sorted[:, 2]
# R_shuffled_90 = R_shuffled_samples_sorted[:, 17]
# R_shuffled_10 = R_shuffled_samples_sorted[:, 2]

# R_shuffled_fivebins = np.load('%sR_sh_plugin_exp_opt_%s_%s_k%d_dmax%d_neuron-%d-%d.npy' %
#                               (dataDir, sampling, rec_length,  N_kappa, dmax_emb_opt, neuron[0], neuron[1]))

"""Plot"""


rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '16.0'
matplotlib.rcParams['xtick.labelsize'] = '16'
matplotlib.rcParams['ytick.labelsize'] = '16'
matplotlib.rcParams['legend.fontsize'] = '16'
matplotlib.rcParams['axes.linewidth'] = 0.6
# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

##### x-axis ####

fig, ((ax1)) = plt.subplots(1, 1, figsize=(3.5, 2.8))

# fig.set_size_inches(4, 3)
ax1.set_xscale('log')
x_min = 0.01
x_max = 10.
ax1.set_xlim((0.01, 10.))
# ax1.set_xticks(np.array([1, 10, 50]))
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.spines['bottom'].set_bounds(0.01, 10.)
ax1.set_xlabel(r'past range $T$ [sec]')

##### y-axis ####
ax1.set_ylabel(r'history dependence $R(T)$')
max_val = np.amax(R_shuffled)
if max_val > 0.2:
    if max_val > 0.3:
        if max_val > 0.405:
            yrange = 0.41
            ymin = 0.1
            ax1.set_ylim((0.1, .51))
            ax1.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
            ax1.spines['left'].set_bounds(.1, 0.5)
        else:
            yrange = 0.3
            ymin = 0.1
            ax1.set_ylim((0.1, .4))
            ax1.set_yticks([0.1, 0.2, 0.3, 0.4])
            ax1.spines['left'].set_bounds(.1, 0.4)
    else:
        yrange = 0.3
        ymin = 0.0
        ax1.set_ylim((0.0, .3))
        ax1.set_yticks([0.0, 0.1, 0.2, 0.3])
        ax1.spines['left'].set_bounds(.0, 0.3)
else:
    yrange = 0.2
    ymin = 0.0
    ax1.set_ylim((0.0, .2))
    ax1.set_yticks([0.0, 0.1, 0.2])
    ax1.spines['left'].set_bounds(.0, 0.2)

##### Unset Borders #####
ax1.spines['top'].set_bounds(0, 0)
ax1.spines['right'].set_bounds(0, 0)
# ax1.plot(Tp_list, M_BBC_uni, linewidth=1.2,
#         label='BBC uniform',  color=soft_red, zorder=2)
# # ax1.fill_between(Tp_list, M_BBC_uni - M_BBC_uni_std,
# #                 M_BBC_uni + M_BBC_uni_std, facecolor=soft_red, alpha=0.3)
# ax1.plot(Tp_list, M_shuffled_uni, linewidth=1.2,
#         label='Shuffle uniform', color=soft_blue, zorder=1)
# # ax1.fill_between(Tp_list, M_shuffled_uni - M_shuffled_uni_std,
# #                 M_shuffled_uni + M_shuffled_uni_std, facecolor=soft_blue, alpha=0.3)
# ax1.plot(Tp_list, R_BBC, linewidth=1.2,  color=main_red,
#          label='BBC', zorder=4)
# ax1.fill_between(Tp_list, R_BBC - R_BBC_std, R_BBC + R_BBC_std,
#                  facecolor=main_red, alpha=0.3)
# ax1.fill_between(Tp_list, R_BBC_10, R_BBC_90,
#                  facecolor=main_red, alpha=0.3)
ax1.plot(Tp_list, R_shuffled, linewidth=1.2, color=green,
         label='Shuffling', zorder=3)
ax1.fill_between(Tp_list, R_shuffled - R_shuffled_std, R_shuffled + R_shuffled_std,
                 facecolor=green, alpha=0.3)
# ax1.fill_between(Tp_list, R_shuffled_10, R_shuffled_90,
#                  facecolor=main_blue, alpha=0.3)

# Compute effective Tdepth and the corresponding history dependence via bootstraps std
# Tp_eff = Tp_list[R_BBC + R_BBC_std >= R_max][0]
# Rtot_eff_fivebins = R_BBC_fivebins[R_BBC_fivebins >= np.amax(
#     R_BBC_fivebins) - R_BBC_iQDist][0]
# Tp_eff_fivebins = Tp_list[R_BBC_fivebins >=
#                           np.amax(R_BBC_fivebins) - R_BBC_iQDist][0]
Rtot_eff_shuffled = R_shuffled[R_shuffled >=
                               np.amax(R_shuffled) - R_shuffled_std[np.argmax(R_BBC)]][0]
Tp_eff_shuffled = Tp_list[R_shuffled >=
                          np.amax(R_shuffled) - R_shuffled_std[np.argmax(R_BBC)]][0]

# ax1.text(0.012, Rtot_eff_BBC + 0.03 * Rtot_eff_BBC, r'$R^*$')
# ax1.text(Tp_eff_BBC + 0.3 * Tp_eff_BBC, .004, r'$T_p^*$')
# fivebins
# ax1.plot([Tp_eff_fivebins], [0], marker='x', color='g',
#          zorder=8)
# ax1.axvline(x=Tp_eff_fivebins, ymax=Rtot_eff_fivebins / 0.2, color='g',
#             linewidth=0.5, linestyle='--')
# x = (np.log10(Tp_eff_fivebins) - np.log10(x_min)) / \
#     (np.log10(x_max) - np.log10(x_min))
# ax1.axhline(y=Rtot_eff_fivebins, xmax=x, color='g',
#             linewidth=0.5, linestyle='--')
# ax1.plot([0.01], [Rtot_eff_fivebins], marker='s', markersize=1.5, color='g',
#          zorder=8)
# ax1.plot([Tp_eff_fivebins], [0.0], marker='s', markersize=1.5, color='g',
#          zorder=8)
# shuffled
ax1.plot([Tp_eff_shuffled], [ymin], marker='x', color=green,
         zorder=8)
ax1.axvline(x=Tp_eff_shuffled, ymax=(Rtot_eff_shuffled - ymin) / yrange, color=green,
            linewidth=0.5, linestyle='--')
x = (np.log10(Tp_eff_shuffled) - np.log10(x_min)) / \
    (np.log10(x_max) - np.log10(x_min))
ax1.axhline(y=Rtot_eff_shuffled, xmax=x, color=green,
            linewidth=0.5, linestyle='--')
ax1.plot([0.01], [Rtot_eff_shuffled], marker='x', color=green,
         zorder=8)
ax1.text(0.012, Rtot_eff_shuffled + 0.03 *
         Rtot_eff_shuffled, r'$\hat{R}_{tot}$')
ax1.text(Tp_eff_shuffled + 0.15 * Tp_eff_shuffled, .005, r'$\hat{T}_D$')

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# fig.savefig("Subsampling_fixed_Cat.pdf")

plt.savefig('../../Paper_Figures/fig4_experimentalResults/Ropt_vs_Tp_neuron_{}-{}.pdf'.format(recorded_system, neuron_str),
            format="pdf", bbox_inches='tight')
plt.savefig('../../Paper_Figures/fig4_experimentalResults/Ropt_vs_Tp_neuron_{}-{}.png'.format(recorded_system, neuron_str),
            format="png", dpi=400, bbox_inches='tight')

plt.show()
plt.close()
