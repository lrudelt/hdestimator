
"""Functions"""
import seaborn.apionly as sns
from scipy.stats import kruskal
from scipy.optimize import bisect
from scipy.io import loadmat
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import matplotlib
import numpy as np
import sys
sys.path.append('../../../Scripts/Functions')
# import plotutils

"""Global Parameters"""
sampling = 'consecutive'
rec_length = '40min'
dmax_emb_opt = 5
neuron_index = 1  # dummy variable
execfile('../../../../Scripts/Paper/params_embedding_opt.py')

# hypothesis test if needed:
# M_median_diff_obs = np.abs(
#     Rtot_Retina_median - Rtot_VisualCortex_median)
# # Create N observations of Null hypothesis
# M_pool = np.append(Rtot_VisualCortex, Rtot_PrefrontalCortex)
# M_median_diff_null = []
# for i in range(100000):
#     M_permut = np.random.permutation(M_pool)
#     M_median_diff_null += [np.median(M_permut[:len(Rtot_PrefrontalCortex)]) -
#                            np.median(M_permut[len(Rtot_PrefrontalCortex):])]
#
# pvalue_permut = len(np.abs(M_median_diff_null)[np.abs(
#     M_median_diff_null) > M_median_diff_obs]) / 100000.
# pvalue_permut

Rtot_eff = {}
Rtot_eff_median = {}
Rtot_eff_median_25 = {}
Rtot_eff_median_975 = {}
Tp_eff = {}
Tp_eff_median = {}
Tp_eff_median_25 = {}
Tp_eff_median_975 = {}

"""Load Neuropixel data"""
for area in ['primaryVisualCortex', 'rostrolateralArea', 'primaryMotorCortex']:
    execfile('../../../../Scripts/Paper/load_data_Neuropixel.py')
    N_neurons = len(recordingNeurons)
    Rtot_eff[area] = np.zeros(N_neurons)
    Tp_eff[area] = np.zeros(N_neurons)
    for i, neuron in enumerate(recordingNeurons):
        R_BBC = np.load('%sR_NSB_exp_opt_%s_%s_k%d_dmax%d_p%d_neuron-%d-%d.npy' %
                        (plottingDataDir, sampling, rec_length,  N_kappa, dmax_emb_opt, p * 100, neuron[0], neuron[1]))
        R_BBC_samples = np.load('%sR_NSB_exp_opt_samples_%s_%s_k%d_dmax%d_p%d_neuron-%d-%d.npy' %
                                (plottingDataDir, sampling_bootstraps, rec_length,  N_kappa, dmax_emb_opt, p * 100, neuron[0], neuron[1]), allow_pickle=True)
        R_BBC_std = np.std(R_BBC_samples, axis=1).astype(float)
        Rtot_eff_BBC = R_BBC[R_BBC >=
                             np.amax(R_BBC) - R_BBC_std[np.argmax(R_BBC)]][0]
        Rtot_eff[area][i] = Rtot_eff_BBC
        Tp_eff[area][i] = Tp_eff_BBC = Tp_list[R_BBC == Rtot_eff_BBC][0] * 1000

    Rtot_eff_median[area] = np.median(Rtot_eff[area])
    Rtot_eff_median_samples = np.sort(np.median(np.random.choice(
        Rtot_eff[area], size=(1000, N_neurons)), axis=1))
    Rtot_eff_median_25[area] = Rtot_eff_median_samples[24] - \
        Rtot_eff_median[area]
    Rtot_eff_median_975[area] = Rtot_eff_median_samples[974] - \
        Rtot_eff_median[area]

    Tp_eff_median[area] = np.median(Tp_eff[area])
    Tp_eff_median_samples = np.sort(np.median(np.random.choice(
        Tp_eff[area], size=(1000, N_neurons)), axis=1))
    Tp_eff_median_25[area] = Tp_eff_median_samples[24] - \
        Tp_eff_median[area]
    Tp_eff_median_975[area] = Tp_eff_median_samples[974] - \
        Tp_eff_median[area]

"""Load Other data"""

for area in ['EC', 'Retina', 'In_vitro']:
    execfile('../../../../Scripts/Paper/load_data_{}.py'.format(area))
    N_neurons = len(recordingNeurons)
    Rtot_eff[area] = np.zeros(N_neurons)
    Tp_eff[area] = np.zeros(N_neurons)
    for i, neuron in enumerate(recordingNeurons):
        R_BBC = np.load('%sR_NSB_exp_opt_%s_%s_k%d_dmax%d_p%d_neuron-%d.npy' %
                        (plottingDataDir, sampling, rec_length,  N_kappa, dmax_emb_opt, p * 100, neuron))
        R_BBC_samples = np.load('%sR_NSB_exp_opt_samples_%s_%s_k%d_dmax%d_p%d_neuron-%d.npy' %
                                (plottingDataDir, sampling_bootstraps, rec_length,  N_kappa, dmax_emb_opt, p * 100, neuron), allow_pickle=True)
        R_BBC_std = np.std(R_BBC_samples, axis=1).astype(float)
        Rtot_eff_BBC = R_BBC[R_BBC >=
                             np.amax(R_BBC) - R_BBC_std[np.argmax(R_BBC)]][0]
        Rtot_eff[area][i] = Rtot_eff_BBC
        Tp_eff[area][i] = Tp_eff_BBC = Tp_list[R_BBC == Rtot_eff_BBC][0] * 1000

    Rtot_eff_median[area] = np.median(Rtot_eff[area])
    Rtot_eff_median_samples = np.sort(np.median(np.random.choice(
        Rtot_eff[area], size=(1000, N_neurons)), axis=1))
    Rtot_eff_median_25[area] = Rtot_eff_median_samples[24] - \
        Rtot_eff_median[area]
    Rtot_eff_median_975[area] = Rtot_eff_median_samples[974] - \
        Rtot_eff_median[area]

    Tp_eff_median[area] = np.median(Tp_eff[area])
    Tp_eff_median_samples = np.sort(np.median(np.random.choice(
        Tp_eff[area], size=(1000, N_neurons)), axis=1))
    Tp_eff_median_25[area] = Tp_eff_median_samples[24] - \
        Tp_eff_median[area]
    Tp_eff_median_975[area] = Tp_eff_median_samples[974] - \
        Tp_eff_median[area]

# fig, ((ax)) = plt.subplots(1, 1, figsize=(2., 3.))
fig, ((ax)) = plt.subplots(1, 1, figsize=(7, 3.2))
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '16.0'
matplotlib.rcParams['xtick.labelsize'] = '16'
matplotlib.rcParams['ytick.labelsize'] = '16'
matplotlib.rcParams['legend.fontsize'] = '16'
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams["errorbar.capsize"] = 2.5

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

##### Unset Borders #####
ax.spines['top'].set_bounds(0, 0)
ax.spines['right'].set_bounds(0, 0)

ax.set_xscale('log')
ax.set_xlim((30, 1000))
# ax.set_xticks(np.array([1, 10, 50]))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.spines['bottom'].set_bounds(30, 1000)
ax.set_xlabel(r'temporal depth $\hat{T}_D$ [ms]')

ax.set_ylabel(
    r'total history dependence $\hat{R}_{tot}$')
ax.set_ylim((0.0, 0.45))
ax.set_yticks([0.0, 0.2, 0.4])
ax.spines['left'].set_bounds(.0, .4)

ax.errorbar(x=[Tp_eff_median['In_vitro']], y=[Rtot_eff_median['In_vitro']],  yerr=[[-Rtot_eff_median_25['In_vitro']], [Rtot_eff_median_975['In_vitro']]], xerr=[[-Tp_eff_median_25['In_vitro']], [Tp_eff_median_975['In_vitro']]],
            color=main_red, marker='v', markersize=6)

ax.errorbar(x=[Tp_eff_median['Retina']], y=[Rtot_eff_median['Retina']],  yerr=[[-Rtot_eff_median_25['Retina']], [Rtot_eff_median_975['Retina']]], xerr=[[-Tp_eff_median_25['Retina']], [Tp_eff_median_975['Retina']]],
            color='orange', marker='o', markersize=6)

ax.errorbar(x=[Tp_eff_median['primaryVisualCortex']], y=[Rtot_eff_median['primaryVisualCortex']],  yerr=[[-Rtot_eff_median_25['primaryVisualCortex']], [Rtot_eff_median_975['primaryVisualCortex']]], xerr=[[-Tp_eff_median_25['primaryVisualCortex']], [Tp_eff_median_975['primaryVisualCortex']]],
            color=green, marker='s', markersize=6)
# rostrolateral
# ax.errorbar(x=[Tp_eff_median['rostrolateralArea']], y=[Rtot_eff_median['rostrolateralArea']],  yerr=[[-Rtot_eff_median_25['rostrolateralArea']], [Rtot_eff_median_975['rostrolateralArea']]], xerr=[[-Tp_eff_median_25['rostrolateralArea']], [Tp_eff_median_975['rostrolateralArea']]],
#             color=green, marker='s', markersize=6)
# # motor
# ax.errorbar(x=[Tp_eff_median['primaryMotorCortex']], y=[Rtot_eff_median['primaryMotorCortex']],  yerr=[[-Rtot_eff_median_25['primaryMotorCortex']], [Rtot_eff_median_975['primaryMotorCortex']]], xerr=[[-Tp_eff_median_25['primaryMotorCortex']], [Tp_eff_median_975['primaryMotorCortex']]],
#             color=violet, marker='s', markersize=6)
# ec
ax.errorbar(x=[Tp_eff_median['EC']], y=[Rtot_eff_median['EC']],  yerr=[[-Rtot_eff_median_25['EC']], [Rtot_eff_median_975['EC']]], xerr=[[-Tp_eff_median_25['EC']], [Tp_eff_median_975['EC']]],
            color=main_blue, marker='D', markersize=6)

ax.scatter(x=[Tp_eff_median['In_vitro']], y=[Rtot_eff_median['In_vitro']],
           color=main_red, marker='v', s=30, label='rat cortical culture')
ax.scatter(x=[Tp_eff_median['Retina']], y=[Rtot_eff_median['Retina']],
           color='orange', marker='o', s=30, label='salamander retina')
ax.scatter(x=[Tp_eff_median['primaryVisualCortex']], y=[Rtot_eff_median['primaryVisualCortex']],
           color=green, marker='s', s=30, label=r'\begin{center}mouse primary \\ visual cortex\end{center}')
ax.scatter(x=[Tp_eff_median['EC']], y=[Rtot_eff_median['EC']],
           color=main_blue, marker='D', s=30, label='rat entorhinal cortex')
# ax.scatter(x=[Tp_eff_median['rostrolateralArea']], y=[Rtot_eff_median['rostrolateralArea']],
#            color=green, marker='s', s=30, label='rostrolateral area')
# ax.scatter(x=[Tp_eff_median['primaryMotorCortex']], y=[Rtot_eff_median['primaryMotorCortex']],
#            color=violet, marker='s', s=30, label='primary motor area')

ax.scatter(Tp_eff['In_vitro'], Rtot_eff['In_vitro'],
           s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
ax.scatter(Tp_eff['Retina'], Rtot_eff['Retina'],
           s=3, color='orange', marker="o", alpha=0.5, zorder=2)
ax.scatter(Tp_eff['primaryVisualCortex'], Rtot_eff['primaryVisualCortex'],
           s=3, color=green, marker="s", alpha=0.5, zorder=2)
# ax.scatter(Tp_eff['rostrolateralArea'], Rtot_eff['rostrolateralArea'],
#            s=3, color=green, marker="s", alpha=0.5, zorder=2)
# ax.scatter(Tp_eff['primaryMotorCortex'], Rtot_eff['primaryMotorCortex'],
#            s=3, color=violet, marker="s", alpha=0.5, zorder=2)
ax.scatter(Tp_eff['EC'], Rtot_eff['EC'],
           s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)

ax.legend(loc=(1.0, 0.3), frameon=False)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# fig.savefig("Subsampling_fixed_Cat.pdf")
# plt.savefig('Poster/Mmax_vs_Tp_scatter.png',
#             format="png", dpi=600, bbox_inches='tight')

plt.savefig('../../Paper_Figures/fig4_experimentalResults/Rtot_vs_Tdepth_scatter.pdf',
            format="pdf", bbox_inches='tight')

plt.show()
plt.close()
