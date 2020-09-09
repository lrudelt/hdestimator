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

"""Load data"""
recorded_system = 'EC'
rec_length = '90min'
number_valid_neurons = 28

R_tot_shuffling_EC = []
R_tot_fivebins_EC = []
R_tot_onebin_EC = []
R_tot_glm_EC = []
for neuron_index in range(number_valid_neurons):
    setup = 'full'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, R_tot_shuffling, T_D_shuffling, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results_full(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_shuffling_EC += [R_tot_shuffling/R_tot_bbc]

    setup = 'fivebins'
    R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results_shuffling(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_fivebins_EC += [R_tot_fivebins/R_tot_bbc]

    setup = 'onebin'
    R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = plots.load_analysis_results_shuffling(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_onebin_EC += [R_tot_onebin/R_tot_bbc]

    R_tot_glm = plots.load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str)
    R_tot_glm_EC += [R_tot_glm/R_tot_bbc]

R_tot_shuffling_EC_median = np.median(R_tot_shuffling_EC)
R_tot_shuffling_EC_median_loCI, R_tot_shuffling_EC_median_hiCI = plots.get_CI_median(R_tot_shuffling_EC)
R_tot_fivebins_EC_median = np.median(R_tot_fivebins_EC)
R_tot_fivebins_EC_median_loCI, R_tot_fivebins_EC_median_hiCI = plots.get_CI_median(R_tot_fivebins_EC)
R_tot_onebin_EC_median = np.median(R_tot_onebin_EC)
R_tot_onebin_EC_median_loCI, R_tot_onebin_EC_median_hiCI = plots.get_CI_median(R_tot_onebin_EC)
R_tot_glm_EC_median = np.median(R_tot_glm_EC)
R_tot_glm_EC_median_loCI, R_tot_glm_EC_median_hiCI = plots.get_CI_median(R_tot_glm_EC)

recorded_system = 'Retina'
rec_length = '90min'
number_valid_neurons = 28

R_tot_shuffling_Retina = []
R_tot_fivebins_Retina = []
R_tot_onebin_Retina = []
R_tot_glm_Retina = []
for neuron_index in range(number_valid_neurons):
    setup = 'full'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, R_tot_shuffling, T_D_shuffling, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results_full(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_shuffling_Retina += [R_tot_shuffling/R_tot_bbc]

    setup = 'fivebins'
    R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results_shuffling(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_fivebins_Retina += [R_tot_fivebins/R_tot_bbc]

    setup = 'onebin'
    R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = plots.load_analysis_results_shuffling(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_onebin_Retina += [R_tot_onebin/R_tot_bbc]

    R_tot_glm = plots.load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str)
    R_tot_glm_Retina += [R_tot_glm/R_tot_bbc]

R_tot_shuffling_Retina_median = np.median(R_tot_shuffling_Retina)
R_tot_shuffling_Retina_median_loCI, R_tot_shuffling_Retina_median_hiCI = plots.get_CI_median(R_tot_shuffling_Retina)
R_tot_fivebins_Retina_median = np.median(R_tot_fivebins_Retina)
R_tot_fivebins_Retina_median_loCI, R_tot_fivebins_Retina_median_hiCI = plots.get_CI_median(R_tot_fivebins_Retina)
R_tot_onebin_Retina_median = np.median(R_tot_onebin_Retina)
R_tot_onebin_Retina_median_loCI, R_tot_onebin_Retina_median_hiCI = plots.get_CI_median(R_tot_onebin_Retina)
R_tot_glm_Retina_median = np.median(R_tot_glm_Retina)
R_tot_glm_Retina_median_loCI, R_tot_glm_Retina_median_hiCI = plots.get_CI_median(R_tot_glm_Retina)

recorded_system = 'Culture'
rec_length = '90min'
number_valid_neurons = 48

R_tot_shuffling_Culture = []
R_tot_fivebins_Culture = []
R_tot_onebin_Culture = []
R_tot_glm_Culture = []
for neuron_index in range(number_valid_neurons):
    setup = 'full'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, R_tot_shuffling, T_D_shuffling, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results_full(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_shuffling_Culture += [R_tot_shuffling/R_tot_bbc]

    setup = 'fivebins'
    R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results_shuffling(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_fivebins_Culture += [R_tot_fivebins/R_tot_bbc]

    setup = 'onebin'
    R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = plots.load_analysis_results_shuffling(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR)
    R_tot_onebin_Culture += [R_tot_onebin/R_tot_bbc]

    R_tot_glm = plots.load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str)
    R_tot_glm_Culture += [R_tot_glm/R_tot_bbc]

R_tot_shuffling_Culture_median = np.median(R_tot_shuffling_Culture)
R_tot_shuffling_Culture_median_loCI, R_tot_shuffling_Culture_median_hiCI = plots.get_CI_median(R_tot_shuffling_Culture)
R_tot_fivebins_Culture_median = np.median(R_tot_fivebins_Culture)
R_tot_fivebins_Culture_median_loCI, R_tot_fivebins_Culture_median_hiCI = plots.get_CI_median(R_tot_fivebins_Culture)
R_tot_onebin_Culture_median = np.median(R_tot_onebin_Culture)
R_tot_onebin_Culture_median_loCI, R_tot_onebin_Culture_median_hiCI = plots.get_CI_median(R_tot_onebin_Culture)
R_tot_glm_Culture_median = np.median(R_tot_glm_Culture)
R_tot_glm_Culture_median_loCI, R_tot_glm_Culture_median_hiCI = plots.get_CI_median(R_tot_glm_Culture)

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]


fig = plt.figure(figsize=(5., 3.))

ax = plt.subplot2grid((17, 1), (0, 0), colspan=1, rowspan=15)
ax2 = plt.subplot2grid((17, 1), (16, 0), colspan=1, rowspan=1, sharex=ax)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_ylabel(
    r'\begin{center} total history dependence $\hat{R}_{\mathrm{tot}}$ \\ relative to BBC \end{center}')
ax.set_ylim((0.55, 1.1))
ax.set_yticks([0.6, 0.8, 1.0])
ax.spines['left'].set_bounds(.55, 1.)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

ax2.set_ylim((0.0, 0.05))
ax2.set_yticks([0.0])
ax2.spines['left'].set_bounds(0, 0.05)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_xticklabels(
    ['EC', 'retina', 'culture'], rotation='horizontal')
# ax2.set_xticks(np.array([1, 10, 50]))
x = [1., 4., 7.]
ax2.set_xlim((-.5, 8.5))
ax2.spines['bottom'].set_bounds(-.5, 8.5)
ax2.set_xticks(x)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

fig.add_subplot(ax)
fig.add_subplot(ax2)
fig.subplots_adjust(hspace=0.1)


rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

# ax1.plt.bar(x=[0.5], height=[np.median(M_BBC_max_In_vivo)],  yerr=[[-M_BBC_max_In_vivo_median_5], [M_BBC_max_In_vivo_median_95]],
# color=main_blue, marker='v', markersize=6, label=r'$R$ model-free')

# EC
ax.bar(x=[0.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax.bar(x=[0.5], height=[R_tot_shuffling_EC_median], yerr=[[R_tot_shuffling_EC_median-R_tot_shuffling_EC_median_loCI], [R_tot_shuffling_EC_median_hiCI-R_tot_shuffling_EC_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax.bar(x=[1.0], height=[R_tot_fivebins_EC_median], yerr=[[R_tot_fivebins_EC_median-R_tot_fivebins_EC_median_loCI], [R_tot_fivebins_EC_median_hiCI-R_tot_fivebins_EC_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax.bar(x=[1.5], height=[R_tot_onebin_EC_median], yerr=[[R_tot_onebin_EC_median-R_tot_onebin_EC_median_loCI], [R_tot_onebin_EC_median_hiCI-R_tot_onebin_EC_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
ax.bar(x=[2.0], height=[R_tot_glm_EC_median], yerr=[[R_tot_glm_EC_median-R_tot_glm_EC_median_loCI], [R_tot_glm_EC_median_hiCI-R_tot_glm_EC_median]], width=.5, alpha=.95,color=violet, ecolor="0.3")

# Retina
ax.bar(x=[3.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax.bar(x=[3.5], height=[R_tot_shuffling_Retina_median], yerr=[[R_tot_shuffling_Retina_median-R_tot_shuffling_Retina_median_loCI], [R_tot_shuffling_Retina_median_hiCI - R_tot_shuffling_Retina_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax.bar(x=[4.0], height=[R_tot_fivebins_Retina_median], yerr=[[R_tot_fivebins_Retina_median-R_tot_fivebins_Retina_median_loCI], [R_tot_fivebins_Retina_median_hiCI -R_tot_fivebins_Retina_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax.bar(x=[4.5], height=[R_tot_onebin_Retina_median], yerr=[[R_tot_onebin_Retina_median-R_tot_onebin_Retina_median_loCI], [R_tot_onebin_Retina_median_hiCI-R_tot_onebin_Retina_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
ax.bar(x=[5.0], height=[R_tot_glm_Retina_median], yerr=[[R_tot_glm_Retina_median-R_tot_glm_Retina_median_loCI], [R_tot_glm_Retina_median_hiCI-R_tot_glm_Retina_median]], width=.5, alpha=.95,color=violet, ecolor="0.3")

# Culture
ax.bar(x=[6.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3", label='BBC')
ax.bar(x=[6.5], height=[R_tot_shuffling_Culture_median], yerr=[[R_tot_shuffling_Culture_median-R_tot_shuffling_Culture_median_loCI], [R_tot_shuffling_Culture_median_hiCI-R_tot_shuffling_Culture_median]], width=.5, alpha=.95,
       color=main_blue, ecolor="0.3", label='Shuffling')
ax.bar(x=[7.0], height=[R_tot_fivebins_Culture_median], yerr=[[R_tot_fivebins_Culture_median-R_tot_fivebins_Culture_median_loCI], [R_tot_fivebins_Culture_median_hiCI-R_tot_fivebins_Culture_median]], width=.5, alpha=.95, color=green, ecolor="0.3", label='max five bins')
ax.bar(x=[7.5], height=[R_tot_onebin_Culture_median], yerr=[[R_tot_onebin_Culture_median-R_tot_onebin_Culture_median_loCI], [R_tot_onebin_Culture_median_hiCI-R_tot_onebin_Culture_median]], width=.5, alpha=.95,color='y', ecolor="0.3", label="single bin")
ax.bar(x=[8.0], height=[R_tot_glm_Culture_median], yerr=[[R_tot_glm_Culture_median-R_tot_glm_Culture_median_loCI], [R_tot_glm_Culture_median_hiCI-R_tot_glm_Culture_median]], width=.5, alpha=.95,color=violet, ecolor="0.3", label='GLM')

number_valid_neurons = 28
ax.scatter(np.zeros(number_valid_neurons) + 0.6, R_tot_shuffling_EC,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 1.1, R_tot_fivebins_EC,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 1.6, R_tot_onebin_EC,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 2.1, R_tot_glm_EC,
           s=3, color="0.7", marker="o", zorder=2)

number_valid_neurons = 28
ax.scatter(np.zeros(number_valid_neurons) + 3.6, R_tot_shuffling_Retina,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 4.1, R_tot_fivebins_Retina,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 4.6, R_tot_onebin_Retina,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 5.1, R_tot_glm_Retina,
           s=3, color="0.7", marker="o", zorder=2)


number_valid_neurons = 48
ax.scatter(np.zeros(number_valid_neurons) + 6.6, R_tot_shuffling_Culture,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 7.1, R_tot_fivebins_Culture,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 7.6, R_tot_onebin_Culture,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 8.1, R_tot_glm_Culture,
           s=3, color="0.7", marker="o", zorder=2)


ax.axhline(y=1, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')
ax.axhline(y=.95, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')
ax.axhline(y=.85, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')
ax.axhline(y=.90, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')
ax.axhline(y=.80, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')

# EC
ax2.bar(x=[0.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax2.bar(x=[0.5], height=[R_tot_shuffling_EC_median], yerr=[[R_tot_shuffling_EC_median-R_tot_shuffling_EC_median_loCI], [R_tot_shuffling_EC_median_hiCI-R_tot_shuffling_EC_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax2.bar(x=[1.0], height=[R_tot_fivebins_EC_median], yerr=[[R_tot_fivebins_EC_median-R_tot_fivebins_EC_median_loCI], [R_tot_fivebins_EC_median_hiCI-R_tot_fivebins_EC_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax2.bar(x=[1.5], height=[R_tot_onebin_EC_median], yerr=[[R_tot_onebin_EC_median-R_tot_onebin_EC_median_loCI], [R_tot_onebin_EC_median_hiCI-R_tot_onebin_EC_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
ax2.bar(x=[2.0], height=[R_tot_glm_EC_median], yerr=[[R_tot_glm_EC_median-R_tot_glm_EC_median_loCI], [R_tot_glm_EC_median_hiCI-R_tot_glm_EC_median]], width=.5, alpha=.95,color=violet, ecolor="0.3")

# Retina
ax2.bar(x=[3.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax2.bar(x=[3.5], height=[R_tot_shuffling_Retina_median], yerr=[[R_tot_shuffling_Retina_median-R_tot_shuffling_Retina_median_loCI], [R_tot_shuffling_Retina_median_hiCI - R_tot_shuffling_Retina_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax2.bar(x=[4.0], height=[R_tot_fivebins_Retina_median], yerr=[[R_tot_fivebins_Retina_median-R_tot_fivebins_Retina_median_loCI], [R_tot_fivebins_Retina_median_hiCI -R_tot_fivebins_Retina_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax2.bar(x=[4.5], height=[R_tot_onebin_Retina_median], yerr=[[R_tot_onebin_Retina_median-R_tot_onebin_Retina_median_loCI], [R_tot_onebin_Retina_median_hiCI-R_tot_onebin_Retina_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
ax2.bar(x=[5.0], height=[R_tot_glm_Retina_median], yerr=[[R_tot_glm_Retina_median-R_tot_glm_Retina_median_loCI], [R_tot_glm_Retina_median_hiCI-R_tot_glm_Retina_median]], width=.5, alpha=.95,color=violet, ecolor="0.3")

# Culture
ax2.bar(x=[6.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3", label='BBC')
ax2.bar(x=[6.5], height=[R_tot_shuffling_Culture_median], yerr=[[R_tot_shuffling_Culture_median-R_tot_shuffling_Culture_median_loCI], [R_tot_shuffling_Culture_median_hiCI-R_tot_shuffling_Culture_median]], width=.5, alpha=.95,
       color=main_blue, ecolor="0.3", label='Shuffling')
ax2.bar(x=[7.0], height=[R_tot_fivebins_Culture_median], yerr=[[R_tot_fivebins_Culture_median-R_tot_fivebins_Culture_median_loCI], [R_tot_fivebins_Culture_median_hiCI-R_tot_fivebins_Culture_median]], width=.5, alpha=.95, color=green, ecolor="0.3", label='max five bins')
ax2.bar(x=[7.5], height=[R_tot_onebin_Culture_median], yerr=[[R_tot_onebin_Culture_median-R_tot_onebin_Culture_median_loCI], [R_tot_onebin_Culture_median_hiCI-R_tot_onebin_Culture_median]], width=.5, alpha=.95,color='y', ecolor="0.3", label="single bin")
ax2.bar(x=[8.0], height=[R_tot_glm_Culture_median], yerr=[[R_tot_glm_Culture_median-R_tot_glm_Culture_median_loCI], [R_tot_glm_Culture_median_hiCI-R_tot_glm_Culture_median]], width=.5, alpha=.95,color=violet, ecolor="0.3", label='GLM')

plt.show()
plt.close()
