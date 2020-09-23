
"""Functions"""
import os
import sys
from sys import exit, stderr, argv, path, modules
from os.path import isfile, isdir, realpath, dirname, exists
from scipy.optimize import bisect
import csv
import yaml
import numpy as np
import pandas as pd
# plotting
import seaborn.apionly as sns
from scipy.optimize import bisect
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import matplotlib

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
recorded_system = "Simulation"


rec_lengths = ['1min', '3min', '5min', '10min', '20min', '45min', '90min']
# Only plot first sample
sample_index = 0

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, (axes) = plt.subplots(7, 2, figsize=(11.5, 19.5))
# fig.set_size_inches(4, 3)

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
soft_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]

"""Load data for all plots"""

# These you should use for now, but they should be updated for publication!
ana_data_path = '/home/lucas/research/projects/history_dependence/analysis/Data/Simulated/analysis_Data'
R_max = np.loadtxt("%s/M_max_5ms.dat" % ana_data_path)
# old T_list
Tmin = 0.01
Tmax = 3.
N_T = 50
T_old = np.zeros(N_T)
kappa_T = bisect(lambda kappa: np.sum(
    Tmin * np.power(10, np.arange(50) * kappa)) - Tmax, -1., 10)
T = 0
for k in range(N_T):
    T += Tmin * np.power(10, k * kappa_T)
    T_old[k] = T
T_old = np.delete(T_old, (23, 34, 42))
R_GLM_BIC = np.delete(np.loadtxt(
    '%s/M_GLM_max.dat' % ana_data_path), (23, 34, 42))

for i, ax in enumerate(axes.flatten()):
    setup_index = i % 2
    rec_length_index = int(i / 2)
    rec_length = rec_lengths[rec_length_index]
    if setup_index == 0:
        bbc_setup = 'full_bbc'
        shuffling_setup = 'full_shuffling'
    else:
        bbc_setup = 'full_bbc_withCV'
        shuffling_setup = 'full_shuffling_withCV'

    """Load data"""
    # Load settings from yaml file

    with open('{}/settings/Simulation_{}.yaml'.format(ESTIMATOR_DIR, bbc_setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)

    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T_bbc, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, sample_index, bbc_setup, ESTIMATOR_DIR, regularization_method = 'bbc')

    R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T_bbc, R_bbc, R_bbc_CI_lo)

    glm_bbc_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_bbc.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    glm_bbc_pd = pd.read_csv(glm_bbc_csv_file_name)
    R_glm_bbc = np.array(glm_bbc_pd['R_GLM'])

    with open('{}/settings/Simulation_{}.yaml'.format(ESTIMATOR_DIR, shuffling_setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)

    ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    analysis_num_str = glm.load_embedding_parameters(
        rec_length, sample_index, analysis_settings)[1]

    R_tot_shuffling, T_D_shuffling, T_shuffling, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, sample_index, shuffling_setup, ESTIMATOR_DIR, regularization_method = 'shuffling')

    R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T_shuffling, R_shuffling, R_shuffling_CI_lo)

    glm_shuffling_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_shuffling.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    glm_shuffling_pd = pd.read_csv(glm_shuffling_csv_file_name)
    R_glm_shuffling = np.array(glm_shuffling_pd['R_GLM'])

    # sns.palplot(sns.color_palette("RdBu_r", 15))  #visualize the color palette
    # ax.set_color_cycle(sns.color_palette("coolwarm_r",num_lines)) setting it
    # as color cycle to do automatised color assignment

    ##########################################
    ########## Simulated Conventional ########
    ##########################################

    ax.set_xscale('log')
    ax.set_xlim((0.01, 3.5))
    x_min = 0.01
    x_max = 3.5
    # ax.set_xticks(np.array([1, 10, 50]))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.spines['bottom'].set_bounds(0.01, 3)
    # ax.set_xlabel(r'memory depth $T_m$ (sec)')

    ##### y-axis ####
    # ax.set_ylabel(r'$M$')
    ax.set_ylim((0.0, .16))
    ax.set_yticks([0.0, 0.08, 0.16])
    yrange = 0.16
    ymin = 0.0
    ax.spines['left'].set_bounds(.0, 0.16)

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)

    if i ==0:
        ax.text(.015, R_tot_shuffling-0.03, r'$\hat{R}_{\mathrm{tot}}$',
                color='0.0', ha='left', va='bottom', fontsize = 20)
        ax.text(T_D_shuffling - 0.45*T_D_shuffling , 0.003, r'$\hat{T}_D$',
                color='0.0', ha='left', va='bottom', fontsize = 20)

    if not int(i/2) == 6:
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        empty_string_labels = ['']*len(xlabels)
        ax.set_xticklabels(empty_string_labels)
    if not i%2 == 0:
        ylabels = [item.get_text() for item in ax.get_yticklabels()]
        empty_string_labels = ['']*len(ylabels)
        ax.set_yticklabels(empty_string_labels)

    # GLM true max and R vs T
    # ax.plot([T[0], T[-1]], [R_max, R_max], '--', color='0.5', zorder=1)
    # ax.plot(T_old, R_GLM_BIC, color='.5', zorder=3)

    # GLM for same embeddings as comparison

    # Embedding optimized estimates and confidence intervals
    ax.plot(T_bbc, R_bbc, linewidth=1.2,  color=main_red, zorder=4, label= 'BBC')
    ax.fill_between(T_bbc, R_bbc_CI_lo, R_bbc_CI_hi, facecolor=main_red, alpha=0.3)
    ax.plot(T_shuffling, R_shuffling, linewidth=1.2, color=main_blue, zorder=3, label= 'Shuffling')
    ax.fill_between(T_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi,
                    facecolor=main_blue, alpha=0.3)

    # Rtot and Tdepth bbc
    ax.plot([T_D_bbc], [ymin], marker='d', markersize = 5., color=main_red,
             zorder=8)
    ax.plot([T_D_bbc], [R_tot_bbc], marker='x', markersize = 6., color=main_red,
             zorder=8)
    ax.axvline(x=T_D_bbc, ymax=(R_tot_bbc - ymin) / yrange, color=main_red,
                linewidth=0.5, linestyle='--')
    x = (np.log10(T_D_bbc) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_bbc, xmax=x, color=main_red,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_bbc], marker='d',markersize = 5., color=main_red,
             zorder=8)

    ax.plot(T_bbc[T_D_index_bbc:max_valid_index_bbc], np.zeros(max_valid_index_bbc-T_D_index_bbc)+R_tot_bbc, color = main_red, linewidth=1.5, linestyle='--')

    # Rtot and Tdepth Shuffling
    ax.plot([T_D_shuffling], [ymin], marker='d', markersize = 5., color=main_blue,
             zorder=8)
    ax.plot([T_D_shuffling], [R_tot_shuffling], marker='x', markersize = 6., color=main_blue,
             zorder=8)
    ax.axvline(x=T_D_shuffling, ymax=(R_tot_shuffling - ymin) / yrange, color=main_blue,
                linewidth=0.5, linestyle='--')
    x = (np.log10(T_D_shuffling) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_shuffling, xmax=x, color=main_blue,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_shuffling], marker='d',markersize = 5., color=main_blue,
             zorder=8)

    ax.plot(T_bbc, R_glm_bbc, '-.', color='.4', alpha=0.8,
            zorder=3, label='true $R(T,d^*,\kappa^*)$ (BBC)')  # , label='Model'
    ax.plot(T_shuffling, R_glm_shuffling, ':', color='.4',
            lw=1.8, alpha=0.8, zorder=2, label=r'true $R(T,d^*,\kappa^*)$ (Shuffling)')

    ax.plot(T_shuffling[T_D_index_shuffling:max_valid_index_shuffling], np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_shuffling, color = main_blue, linewidth=1.5, linestyle='--')

    # ax.axvline(x=T_D_shuffling, ymax=0.9, color=main_blue,
    #            linewidth=0.5, linestyle='--')
    # ax.axhline(y=R_tot_shuffling, xmax=.7, color=main_blue,
    #            linewidth=0.5, linestyle='--')
    # ax.plot([0.01], [R_tot_shuffling], marker='d', markersize=3, color=main_blue,
    #         zorder=8)
    # ax.plot([T_D_shuffling], [0.0], marker='d', markersize=3, color=main_blue,
    #         zorder=8)
    # ax.plot([T_D_shuffling], [R_tot_shuffling], marker='x', markersize=6, color=main_blue,
    #         zorder=8)
    if setup_index == 0:
        ax.set_title('{}, no cross-validation'.format(rec_length))
    else:
        ax.set_title('{}, with cross-validation'.format(rec_length))
    if i ==0:
        ax.legend(loc=(0.6, 0.0), frameon=False)
# ax.text(0.012, M_max + 0.02 * M_max, r'$\hat{R}_{tot}$')
# ax.text(T_D_shuffling + 0.15 * Tm_eff, .101, r'$\hat{T}_D$')
fig.text(0.5, - 0.01, r'past range $T$ [sec]', ha='center', va='center', fontsize = 20)
fig.text(-0.01, 0.5, r'history dependence $R(T)$', ha='center', va='center', rotation='vertical',  fontsize = 20)


fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# fig.savefig("Subsampling_fixed_Cat.pdf")
plt.savefig('{}/Ropt_vs_T_Simulation_examples.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
# plt.savefig('../Mopt_vs_Tm_model_comparison.png',
#             format="png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
