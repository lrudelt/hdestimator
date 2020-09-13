"""Functions"""
import seaborn.apionly as sns
from scipy.optimize import bisect
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import matplotlib
import numpy as np
from sys import exit, stderr, argv, path, modules
import pandas as pd
import yaml
# import plotutils

# ESTIMATOR_DIR = '{}/../..'.format(dirname(realpath(__file__)))
ESTIMATOR_DIR = '/home/lucas/research/projects/history_dependence/hdestimator'
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))
if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots

# fig = dirname(realpath(__file__)).split("/")[-1]
fig = 'fig4'
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)

"""Global parameters"""
setup = 'fivebins'
rec_length = '40min'
k = 5

"""Load data"""
recorded_system = 'EC'
number_valid_neurons = 28
R_tot_EC = []
T_D_EC = []
R_tot_new_EC = []
T_D_new_EC = []
for neuron_index in range(number_valid_neurons):
    R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
    R_tot_EC += [R_tot]
    T_D_EC += [T_D*1000]
    # T_D_new, R_tot_new, R_tot_std_new, T_D_index, max_valid_index = plots.get_temporal_depth_and_R_tot(T, R)
    # T_D_new_EC += [T_D_new*1000]
    # R_tot_new_EC += [R_tot_new]

R_tot_EC_median = np.median(R_tot_EC)
R_tot_EC_median_loCI, R_tot_EC_median_hiCI = plots.get_CI_median(R_tot_EC)
T_D_EC_median = np.median(T_D_EC)
T_D_EC_median_loCI, T_D_EC_median_hiCI = plots.get_CI_median(T_D_EC)
# R_tot_EC_median = np.median(R_tot_new_EC)
# R_tot_EC_median_loCI, R_tot_EC_median_hiCI = plots.get_CI_median(R_tot_new_EC)
# T_D_EC_median = np.median(T_D_new_EC)
# T_D_EC_median_loCI, T_D_EC_median_hiCI = plots.get_CI_median(T_D_new_EC)

recorded_system = 'Retina'
number_valid_neurons = 111
R_tot_Retina = []
T_D_Retina = []
R_tot_new_Retina = []
T_D_new_Retina = []
for neuron_index in range(number_valid_neurons):
    R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
    R_tot_Retina += [R_tot]
    T_D_Retina += [T_D*1000]
    # T_D_new, R_tot_new, R_tot_std_new, T_D_index, max_valid_index = plots.get_temporal_depth_and_R_tot(T, R)
    # T_D_new_Retina += [T_D_new*1000]
    # R_tot_new_Retina += [R_tot_new]

R_tot_Retina_median = np.median(R_tot_Retina)
R_tot_Retina_median_loCI, R_tot_Retina_median_hiCI = plots.get_CI_median(R_tot_Retina)
T_D_Retina_median = np.median(T_D_Retina)
T_D_Retina_median_loCI, T_D_Retina_median_hiCI = plots.get_CI_median(T_D_Retina)
# R_tot_Retina_median = np.median(R_tot_new_Retina)
# R_tot_Retina_median_loCI, R_tot_Retina_median_hiCI = plots.get_CI_median(R_tot_new_Retina)
# T_D_Retina_median = np.median(T_D_new_Retina)
# T_D_Retina_median_loCI, T_D_Retina_median_hiCI = plots.get_CI_median(T_D_new_Retina)

recorded_system = 'Culture'
number_valid_neurons = 48
R_tot_Culture = []
T_D_Culture = []
R_tot_new_Culture = []
T_D_new_Culture = []
for neuron_index in range(number_valid_neurons):
    R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
    R_tot_Culture += [R_tot]
    T_D_Culture += [T_D*1000]
    # T_D_new, R_tot_new, R_tot_std_new, T_D_index, max_valid_index = plots.get_temporal_depth_and_R_tot(T, R)
    # T_D_new_Culture += [T_D_new*1000]
    # R_tot_new_Culture += [R_tot_new]

R_tot_Culture_median = np.median(R_tot_Culture)
R_tot_Culture_median_loCI, R_tot_Culture_median_hiCI = plots.get_CI_median(R_tot_Culture)
T_D_Culture_median = np.median(T_D_Culture)
T_D_Culture_median_loCI, T_D_Culture_median_hiCI = plots.get_CI_median(T_D_Culture)
# R_tot_Culture_median = np.median(R_tot_new_Culture)
# R_tot_Culture_median_loCI, R_tot_Culture_median_hiCI = plots.get_CI_median(R_tot_new_Culture)
# T_D_Culture_median = np.median(T_D_new_Culture)
# T_D_Culture_median_loCI, T_D_Culture_median_hiCI = plots.get_CI_median(T_D_new_Culture)

recorded_system = 'V1'
number_valid_neurons = 142
R_tot_V1 = []
T_D_V1 = []
R_tot_new_V1 = []
T_D_new_V1 = []
for neuron_index in range(number_valid_neurons):
    R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
    R_tot_V1 += [R_tot]
    T_D_V1 += [T_D*1000]
    # T_D_new, R_tot_new, R_tot_std_new, T_D_index, max_valid_index = plots.get_temporal_depth_and_R_tot(T, R)
    # T_D_new_V1 += [T_D_new*1000]
    # R_tot_new_V1 += [R_tot_new]

R_tot_V1_median = np.median(R_tot_V1)
R_tot_V1_median_loCI, R_tot_V1_median_hiCI = plots.get_CI_median(R_tot_V1)
T_D_V1_median = np.median(T_D_V1)
T_D_V1_median_loCI, T_D_V1_median_hiCI = plots.get_CI_median(T_D_V1)
# R_tot_V1_median = np.median(R_tot_new_V1)
# R_tot_V1_median_loCI, R_tot_V1_median_hiCI = plots.get_CI_median(R_tot_new_V1)
# T_D_V1_median = np.median(T_D_new_V1)
# T_D_V1_median_loCI, T_D_V1_median_hiCI = plots.get_CI_median(T_D_new_V1)

# fig, ((ax)) = plt.subplots(1, 1, figsize=(2., k.))
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
    r'total history dependence $\hat{R}_{\mathrm{tot}}$')
ax.set_ylim((0.0, 0.45))
ax.set_yticks([0.0, 0.2, 0.4])
ax.spines['left'].set_bounds(.0, .4)

ax.errorbar(x=[T_D_Culture_median], y=[R_tot_Culture_median],  yerr=[[R_tot_Culture_median-R_tot_Culture_median_loCI], [R_tot_Culture_median_hiCI-R_tot_Culture_median]], xerr=[[T_D_Culture_median-T_D_Culture_median_loCI], [T_D_Culture_median_hiCI-T_D_Culture_median]], color=main_red, marker='v', markersize=6)

ax.errorbar(x=[T_D_Retina_median], y=[R_tot_Retina_median],  yerr=[[R_tot_Retina_median-R_tot_Retina_median_loCI], [R_tot_Retina_median_hiCI-R_tot_Retina_median]], xerr=[[T_D_Retina_median-T_D_Retina_median_loCI], [T_D_Retina_median_hiCI-T_D_Retina_median]], color='orange', marker='o', markersize=6)

ax.errorbar(x=[T_D_V1_median], y=[R_tot_V1_median],  yerr=[[R_tot_V1_median-R_tot_V1_median_loCI], [R_tot_V1_median_hiCI-R_tot_V1_median]], xerr=[[T_D_V1_median-T_D_V1_median_loCI], [T_D_V1_median_hiCI-T_D_V1_median]], color=green, marker='s', markersize=6)

ax.errorbar(x=[T_D_EC_median], y=[R_tot_EC_median],  yerr=[[R_tot_EC_median-R_tot_EC_median_loCI], [R_tot_EC_median_hiCI-R_tot_EC_median]], xerr=[[T_D_EC_median-T_D_EC_median_loCI], [T_D_EC_median_hiCI-T_D_EC_median]], color=main_blue, marker='D', markersize=6)

ax.scatter(x=[T_D_Culture_median], y=[R_tot_Culture_median],
           color=main_red, marker='v', s=30, label='rat cortical culture')
ax.scatter(x=[T_D_Retina_median], y=[R_tot_Retina_median],
           color='orange', marker='o', s=30, label='salamander retina')
ax.scatter(x=[T_D_V1_median], y=[R_tot_V1_median],
           color=green, marker='s', s=30, label=r'\begin{center}mouse primary \\ visual cortex\end{center}')
ax.scatter(x=[T_D_EC_median], y=[R_tot_EC_median],
           color=main_blue, marker='D', s=30, label='rat entorhinal cortex')
# ax.scatter(x=[Tp_eff_median['rostrolateralArea']], y=[Rtot_eff_median['rostrolateralArea']],
#            color=green, marker='s', s=30, label='rostrolateral area')
# ax.scatter(x=[Tp_eff_median['primaryMotorCortex']], y=[Rtot_eff_median['primaryMotorCortex']],
#            color=violet, marker='s', s=30, label='primary motor area')

ax.scatter(T_D_Culture, R_tot_Culture,
           s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
ax.scatter(T_D_Retina, R_tot_Retina,
           s=3, color='orange', marker="o", alpha=0.5, zorder=2)
ax.scatter(T_D_V1, R_tot_V1,
           s=3, color=green, marker="s", alpha=0.5, zorder=2)
ax.scatter(T_D_EC, R_tot_EC,
           s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)

ax.legend(loc=(1.0, 0.3), frameon=False)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# fig.savefig("Subsampling_fixed_Cat.pdf")
# plt.savefig('Poster/Mmax_vs_Tp_scatter.png',
#             format="png", dpi=600, bbox_inches='tight')
#
# plt.savefig('{}/Rtot_vs_Tdepth_scatter.pdf'.format(PLOTTING_DIR),
#             format="pdf", bbox_inches='tight')

plt.show()
plt.close()
