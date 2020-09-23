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
fig = 'supplementaries'
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)

"""Parameters and Settings"""
recorded_system = 'Retina'
rec_length = '90min'
dataDir = '/data.nst/lucas/history_dependence/paper/retina_data/'
validNeurons = np.load('{}validNeurons.npy'.format(dataDir)).astype(int)

"""Plotting"""
# Font
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

fig, axes = plt.subplots(16, 7, figsize=(14., 19.5))

# Sort neurons, put neurons with max_val > 0.2 and max_val <0.3 in a separate group

smallR = []
mediumR = []
highR = []
veryhighR = []
setup = 'full_shuffling'
for neuron_index, neuron in enumerate(validNeurons):
    R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    max_val = np.amax(R)
    if max_val > 0.2:
        if max_val >0.3:
            if max_val > 0.405:
                veryhighR += [neuron_index]
            else:
                highR += [neuron_index]
        else:
            mediumR+= [neuron_index]
    else:
        smallR += [neuron_index]

index_small_to_medium = len(smallR)
index_medium_to_high = len(smallR)+len(mediumR)
index_high_to_veryhigh = len(smallR)+len(mediumR)+len(highR)

# two rows 0, 2
# three rows high
#

for k, neuron_index in enumerate(np.append(np.append(np.append(smallR, mediumR),highR),veryhighR).astype(int)):
    # print(neuron_index)
    ax = axes[int(k/7)][k%7]

    """Load data full"""
    setup = 'full_bbc'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T_bbc, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'bbc')
    # T_D_bbc, R_tot_bbc, R_tot_std_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_temporal_depth_and_R_tot(T, R_bbc)
    R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T_bbc, R_bbc, R_bbc_CI_lo)
    max_index = np.argmax(R_bbc)
    for i, R_val in enumerate(R_bbc):
        if not i == max_index:
            R_bbc_CI_lo[i] = R_val
            R_bbc_CI_hi[i] = R_val
    # Get R_tot_glm for T_D
    R_tot_glm = plots.load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str)

    setup = 'full_shuffling'
    R_tot_shuffling, T_D_shuffling, T_shuffling, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    # T_D_shuffling, R_tot_shuffling, R_tot_std_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_temporal_depth_and_R_tot(T, R_shuffling)
    R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T_shuffling, R_shuffling, R_shuffling_CI_lo)
    max_index = np.argmax(R_shuffling)
    for i, R_val in enumerate(R_shuffling):
        if not i == max_index:
            R_shuffling_CI_lo[i] = R_val
            R_shuffling_CI_hi[i] = R_val

    """Load data five bins"""
    setup = 'fivebins'
    R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    # T_D_fivebins, R_tot_fivebins, R_tot_std_fivebins, T_D_index_fivebins, max_valid_index_fivebins = plots.get_temporal_depth_and_R_tot(T, R_fivebins)
    R_tot_fivebins, T_D_index_fivebins, max_valid_index_fivebins = plots.get_R_tot(T, R_fivebins, R_fivebins_CI_lo)
    for i, R_val in enumerate(R_fivebins):
        if not i == max_index:
            R_fivebins_CI_lo[i] = R_val
            R_fivebins_CI_hi[i] = R_val

    """Load data onebins"""
    setup = 'onebin'
    R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    # T_D_onebin, R_tot_onebin, R_tot_std_onebin, T_D_index_onebin, max_valid_index_onebin = plots.get_temporal_depth_and_R_tot(T, R_onebin)
    R_tot_onebin, T_D_index_onebin, max_valid_index_onebin = plots.get_R_tot(T, R_onebin, R_onebin_CI_lo)
    for i, R_val in enumerate(R_onebin):
        if not i == max_index:
            R_onebin_CI_lo[i] = R_val
            R_onebin_CI_hi[i] = R_val

    ax.set_xscale('log')
    x_min = 0.005
    x_max = 5.
    ax.set_xlim((0.005, 5.))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.spines['bottom'].set_bounds(0.005, 5.)
    ax.set_xticks([0.01, 0.1, 1.0])

    ##### y-axis ####
    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    max_val = np.amax(R_shuffling)
    if max_val > 0.2:
        if max_val > 0.3:
            if max_val > 0.405:
                yrange = 0.41
                ymin = 0.1
                ax.set_ylim((0.1, .51))
                ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
                ax.spines['left'].set_bounds(.1, 0.5)
            else:
                yrange = 0.3
                ymin = 0.1
                ax.set_ylim((0.1, .4))
                ax.set_yticks([0.1, 0.2, 0.3, 0.4])
                ax.spines['left'].set_bounds(.1, 0.4)
        else:
            yrange = 0.3
            ymin = 0.0
            ax.set_ylim((0.0, .3))
            ax.set_yticks([0.0, 0.1, 0.2, 0.3])
            ax.spines['left'].set_bounds(.0, 0.3)
    else:
        yrange = 0.2
        ymin = 0.0
        ax.set_ylim((0.0, .2))
        ax.set_yticks([0.0, 0.1, 0.2])
        ax.spines['left'].set_bounds(.0, 0.2)

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)

    """BBC"""
    ax.plot(T_bbc, R_bbc, linewidth=1.2, color=main_red,
             label=r'BBC, $d_{\mathrm{max}}=20$', zorder=10)
    ax.fill_between(T_bbc, R_bbc_CI_lo, R_bbc_CI_hi,
                    facecolor=main_red, zorder= 10, alpha=0.3)

    ax.plot([T_D_bbc], [ymin], marker='d', markersize = 5., color=main_red,
             zorder=8)
    ax.plot([T_D_bbc], [R_tot_bbc], marker='x', markersize = 6., color=main_red,
             zorder=14)
    ax.axvline(x=T_D_bbc, ymax=(R_tot_bbc - ymin) / yrange, color=main_red,
                linewidth=0.5, linestyle='--')
    x = (np.log10(T_D_bbc) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_bbc, xmax=x, color=main_red,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_bbc], marker='d',markersize = 5., color=main_red,
             zorder=8)
    if k == 0:
        ax.text(0.007, R_tot_bbc + 0.04 *
                 R_tot_bbc, r'$\hat{R}_{\mathrm{tot}}$')
        ax.text(T_D_bbc + 0.7 * T_D_bbc, ymin + .005, r'$\hat{T}_D$')

    ax.plot(T[T_D_index_bbc:max_valid_index_bbc], np.zeros(max_valid_index_bbc-T_D_index_bbc)+R_tot_bbc, color = main_red, linewidth=1.5, linestyle='--')
    # R_tot_CI_lo = R_tot_bbc - 2 *  R_tot_std_bbc
    # R_tot_CI_hi = R_tot_bbc + 2 *  R_tot_std_bbc
    # ax.fill_between(T[T_D_index_bbc:max_valid_index_bbc], np.zeros(max_valid_index_bbc-T_D_index_bbc)+R_tot_CI_lo, np.zeros(max_valid_index_bbc-T_D_index_bbc)+R_tot_CI_hi, facecolor=main_red, alpha=0.3)


    """Shuffling"""
    ax.plot(T, R_shuffling, linewidth=1.2, color=main_blue,
             label=r'Shuffling, $d_{\mathrm{max}}=20$', zorder=3)
    ax.fill_between(T, R_shuffling_CI_lo, R_shuffling_CI_hi,
                facecolor=main_blue, zorder= 8, alpha=0.3)
    ax.plot([T_D_shuffling], [ymin], marker='d', markersize = 5., color=main_blue,
             zorder=8)
    ax.plot([T_D_shuffling], [R_tot_shuffling], marker='x', markersize = 6., color=main_blue,
             zorder=13)
    ax.axvline(x=T_D_shuffling, ymax=(R_tot_shuffling - ymin) / yrange, color=main_blue,
                linewidth=0.5, linestyle='--')
    x = (np.log10(T_D_shuffling) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_shuffling, xmax=x, color=main_blue,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_shuffling], marker='d',markersize = 5., color=main_blue,
             zorder=8)

    ax.plot(T[T_D_index_shuffling:max_valid_index_shuffling], np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_shuffling, color = main_blue, linewidth=1.5, linestyle='--')
    # R_tot_CI_lo = R_tot_shuffling - 2 *  R_tot_std_shuffling
    # R_tot_CI_hi = R_tot_shuffling + 2 *  R_tot_std_shuffling
    # ax.fill_between(T[T_D_index_shuffling:max_valid_index_shuffling], np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_CI_lo, np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_CI_hi, facecolor=main_blue, alpha=0.3)

    """Fivebins"""
    ax.plot(T, R_fivebins, linewidth=1.2, color=green,
             label=r'Shuffling, $d_{\mathrm{max}}=5$', zorder=3)
    ax.fill_between(T, R_fivebins_CI_lo, R_fivebins_CI_hi,
                    facecolor=green, zorder= 10, alpha=0.3)
    ax.plot([T_D_fivebins], [ymin], marker='d', markersize = 5., color=green,
             zorder=8)
    ax.plot([T_D_fivebins], [R_tot_fivebins], marker='x', markersize = 6., color=green,
             zorder=12)
    ax.axvline(x=T_D_fivebins, ymax=(R_tot_fivebins - ymin) / yrange, color=green,
                linewidth=0.5, linestyle='--')
    x = (np.log10(T_D_fivebins) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_fivebins, xmax=x, color=green,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_fivebins], marker='d',markersize = 5., color=green,
             zorder=8)

    ax.plot(T[T_D_index_fivebins:max_valid_index_fivebins], np.zeros(max_valid_index_fivebins-T_D_index_fivebins)+R_tot_fivebins, color = green, linewidth=1.5, linestyle='--')
    # R_tot_CI_lo = R_tot_fivebins - 2 *  R_tot_std_fivebins
    # R_tot_CI_hi = R_tot_fivebins + 2 *  R_tot_std_fivebins
    # ax.fill_between(T[T_D_index_fivebins:max_valid_index_fivebins], np.zeros(max_valid_index_fivebins-T_D_index_fivebins)+R_tot_CI_lo, np.zeros(max_valid_index_fivebins-T_D_index_fivebins)+R_tot_CI_hi, facecolor=green, alpha=0.3)

    """One bin"""
    ax.plot(T, R_onebin, linewidth=1.2, color='y',
             label=r'Shuffling, $d_{\mathrm{max}}=1$', zorder=3)
    ax.fill_between(T, R_onebin_CI_lo, R_onebin_CI_hi,
                    facecolor='y', zorder= 10, alpha=0.3)
    ax.plot([T_D_onebin], [ymin], marker='d', markersize = 5., color='y',
             zorder=8)
    ax.plot([T_D_onebin], [R_tot_onebin], marker='x', markersize = 6., color='y',
             zorder=8)
    ax.axvline(x=T_D_onebin, ymax=(R_tot_onebin - ymin) / yrange, color='y',
                linewidth=0.5, linestyle='--')
    x = (np.log10(T_D_onebin) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_onebin, xmax=x, color='y',
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_onebin], marker='d',markersize = 5., color='y',
             zorder=8)

    ax.plot(T[T_D_index_onebin:max_valid_index_onebin], np.zeros(max_valid_index_onebin-T_D_index_onebin)+R_tot_onebin, color = 'y', linewidth=1.5, linestyle='--')
    # R_tot_CI_lo = R_tot_onebin - 2 *  R_tot_std_onebin
    # R_tot_CI_hi = R_tot_onebin + 2 *  R_tot_std_onebin
    # ax.fill_between(T[T_D_index_onebin:max_valid_index_onebin], np.zeros(max_valid_index_onebin-T_D_index_onebin)+R_tot_CI_lo, np.zeros(max_valid_index_onebin-T_D_index_onebin)+R_tot_CI_hi, facecolor='y', alpha=0.3)


    """GLM"""
    # Plot R_tot_glm
    ax.plot([T_D_bbc], [R_tot_glm], 's', color=violet, label=r'GLM, $d_{\mathrm{max}}=50$')

    if not int(k/7) == 15:
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        empty_string_labels = ['']*len(xlabels)
        ax.set_xticklabels(empty_string_labels)
    if not k%7 == 0:
        if not k == index_small_to_medium:
            if not k == index_medium_to_high:
                if not k == index_high_to_veryhigh:
                    ylabels = [item.get_text() for item in ax.get_yticklabels()]
                    empty_string_labels = ['']*len(ylabels)
                    ax.set_yticklabels(empty_string_labels)
    if k == 0:
        ax.legend(loc=(-0.1, 1.1), frameon=False)


ax = axes[15][6]
ax.spines['top'].set_bounds(0, 0)
ax.spines['right'].set_bounds(0, 0)
ax.spines['bottom'].set_bounds(0, 0)
ax.spines['left'].set_bounds(0, 0)
ax.set_xticks([])
ax.set_yticks([])

fig.text(0.5, - 0.01, r'past range $T$ [sec]', ha='center', va='center', fontsize = 17)
fig.text(-0.01, 0.5, r'history dependence $R(T)$', ha='center', va='center', rotation='vertical',  fontsize = 17)

fig.tight_layout(pad=0.00, w_pad=-2, h_pad=1)
plt.savefig('{}/Ropt_vs_T_all_Retina.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')

plt.show()
plt.close()
