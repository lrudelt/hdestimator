
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
from matplotlib.ticker import NullFormatter

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

rec_lengths = ['1min', '3min', '5min', '10min', '20min', '45min', '90min']
rec_length_values = [60., 180., 300., 600., 1200., 2700., 5400.]
rec_lengths_colors_bbc = [sns.color_palette("RdBu_r", 15)[8], sns.color_palette("RdBu_r", 15)[9], sns.color_palette("RdBu_r", 15)[10], sns.color_palette(
    "RdBu_r", 15)[11], sns.color_palette("RdBu_r", 15)[12], sns.color_palette("RdBu_r", 15)[13], sns.color_palette("RdBu_r", 15)[14]]
setups = ['full', 'full_withCV']
# Only plot first sample

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6


# Colors
rec_length_colors = {'bbc-1min': sns.color_palette("RdBu_r", 16)[9],
                     'bbc-3min': sns.color_palette("RdBu_r", 16)[10],
                     'bbc-5min': sns.color_palette("RdBu_r", 16)[11],
                     'bbc-10min': sns.color_palette("RdBu_r", 16)[12],
                     'bbc-20min': sns.color_palette("RdBu_r", 16)[13],
                     'bbc-45min': sns.color_palette("RdBu_r", 16)[14],
                     'bbc-90min': sns.color_palette("RdBu_r", 16)[15],
                     'shuffling-1min': sns.color_palette("RdBu_r", 16)[6],
                     'shuffling-3min': sns.color_palette("RdBu_r", 16)[5],
                     'shuffling-5min': sns.color_palette("RdBu_r", 16)[4],
                     'shuffling-10min': sns.color_palette("RdBu_r", 16)[3],
                     'shuffling-20min': sns.color_palette("RdBu_r", 16)[2],
                     'shuffling-45min': sns.color_palette("RdBu_r", 16)[1],
                     'shuffling-90min': sns.color_palette("RdBu_r", 16)[0]}


"""Load data for all plots"""
median_rel_bias_lowT = {}
median_CI_rel_bias_lowT = {}
median_rel_bias_mediumT = {}
median_CI_rel_bias_mediumT = {}
median_rel_bias_highT = {}
median_CI_rel_bias_highT = {}
median_rel_bias_T_D = {}
median_CI_rel_bias_T_D = {}
mean_rel_bias_T_D = {}
mean_CI_rel_bias_T_D = {}

std_rel_bias = {}
for setup in setups:
    # Load settings from yaml file
    with open('{}/settings/Simulation_{}.yaml'.format(ESTIMATOR_DIR, setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)
    ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    T = np.array(analysis_settings['embedding_past_range_set']).astype(float)
    for rec_length in rec_lengths:
        rel_bias_bbc = []
        rel_bias_bbc_lowT = []
        rel_bias_bbc_mediumT = []
        rel_bias_bbc_highT = []
        rel_bias_bbc_T_D = []
        rel_bias_shuffling = []
        rel_bias_shuffling_lowT = []
        rel_bias_shuffling_mediumT = []
        rel_bias_shuffling_highT = []
        rel_bias_shuffling_T_D = []
        number_samples = 30
        if rec_length == '45min':
            number_samples = 10
        if rec_length == '90min':
            number_samples = 10
        for sample_index in range(number_samples):
            analysis_num_str = glm.load_embedding_parameters(
                rec_length, sample_index, analysis_settings)[2]
            # Embedding optimized estimates and confidence intervals
            # hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
            #     ANALYSIS_DIR, analysis_num_str)
            # hisdep_pd = pd.read_csv(hisdep_csv_file_name)
            # R_bbc = np.array(hisdep_pd['max_R_bbc'])
            # R_shuffling = np.array(hisdep_pd['max_R_shuffling'])
            # T = np.array(hisdep_pd['#T'])

            # Temporal depth and total history dependence
            statistics_csv_file_name = '{}/ANALYSIS{}/statistics.csv'.format(
                ANALYSIS_DIR, analysis_num_str)
            statistics_pd = pd.read_csv(statistics_csv_file_name)

            glm_bbc_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_bbc.csv'.format(
                ANALYSIS_DIR, analysis_num_str)
            glm_bbc_pd = pd.read_csv(glm_bbc_csv_file_name)
            T = np.array(glm_bbc_pd['T'])
            R_glm_bbc = np.array(glm_bbc_pd['R_GLM'])

            # TODO: Recompute T based on T in glm benchmark file to load R_tot_glm_bbc
            R_tot_bbc = statistics_pd['R_tot_bbc'][0]
            T_D_bbc = statistics_pd['T_D_bbc'][0]
            T_index = np.where(T == T_D_bbc)[0][0]
            R_tot_glm_bbc = R_glm_bbc[T_index]

            glm_shuffling_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_shuffling.csv'.format(
                ANALYSIS_DIR, analysis_num_str)
            glm_shuffling_pd = pd.read_csv(glm_shuffling_csv_file_name)
            T = np.array(glm_shuffling_pd['T'])
            R_glm_shuffling = np.array(glm_shuffling_pd['R_GLM'])

            R_tot_shuffling = statistics_pd['R_tot_shuffling'][0]
            T_D_shuffling = statistics_pd['T_D_shuffling'][0]
            T_index = np.where(T == T_D_shuffling)[0][0]
            R_tot_glm_shuffling = R_glm_shuffling[T_index]

            # rel_bias_bbc += [100 * (R_bbc - R_glm_bbc) / R_glm_bbc]
            # rel_bias_shuffling += [100 * (R_shuffling -
            #                               R_glm_shuffling) / R_glm_shuffling]
            # rel_bias_bbc_lowT += [100 *
            #                       (R_bbc[10] - R_glm_bbc[10]) / R_glm_bbc[10]]
            # rel_bias_bbc_mediumT += [100 *
            #                          (R_bbc[30] - R_glm_bbc[30]) / R_glm_bbc[30]]
            # rel_bias_bbc_highT += [100 *
            #                        (R_bbc[50] - R_glm_bbc[50]) / R_glm_bbc[50]]
            rel_bias_bbc_T_D += [100 *
                                 (R_tot_bbc - R_tot_glm_bbc) / R_tot_glm_bbc]
            # rel_bias_shuffling_lowT += [
            #     100 * (R_shuffling[10] - R_glm_shuffling[10]) / R_glm_shuffling[10]]
            # rel_bias_shuffling_mediumT += [
            #     100 * (R_shuffling[30] - R_glm_shuffling[30]) / R_glm_shuffling[30]]
            # rel_bias_shuffling_highT += [
            #     100 * (R_shuffling[50] - R_glm_shuffling[50]) / R_glm_shuffling[50]]
            rel_bias_shuffling_T_D += [100 *
                                       (R_tot_shuffling - R_tot_glm_shuffling) / R_tot_glm_shuffling]
        # median_rel_bias_lowT['bbc-{}-{}'.format(setup, rec_length)
        #                      ] = np.median(rel_bias_bbc_lowT)
        # median_CI_rel_bias_lowT['bbc-{}-{}'.format(setup, rec_length)
        #                         ] = plots.get_CI_median(rel_bias_bbc_lowT)
        # median_rel_bias_lowT['shuffling-{}-{}'.format(setup, rec_length)
        #                      ] = np.median(rel_bias_shuffling_lowT)
        # median_CI_rel_bias_lowT['shuffling-{}-{}'.format(setup, rec_length)
        #                         ] = plots.get_CI_median(rel_bias_shuffling_lowT)
        # median_rel_bias_mediumT['bbc-{}-{}'.format(setup, rec_length)
        #                         ] = np.median(rel_bias_bbc_mediumT)
        # median_CI_rel_bias_mediumT['bbc-{}-{}'.format(setup, rec_length)
        #                            ] = plots.get_CI_median(rel_bias_bbc_mediumT)
        # median_rel_bias_mediumT['shuffling-{}-{}'.format(setup, rec_length)
        #                         ] = np.median(rel_bias_shuffling_mediumT)
        # median_CI_rel_bias_mediumT['shuffling-{}-{}'.format(setup, rec_length)
        #                            ] = plots.get_CI_median(rel_bias_shuffling_mediumT)
        # median_rel_bias_highT['bbc-{}-{}'.format(setup, rec_length)
        #                       ] = np.median(rel_bias_bbc_highT)
        # median_CI_rel_bias_highT['bbc-{}-{}'.format(setup, rec_length)
        #                          ] = plots.get_CI_median(rel_bias_bbc_highT)
        # median_rel_bias_highT['shuffling-{}-{}'.format(setup, rec_length)
        #                       ] = np.median(rel_bias_shuffling_highT)
        # median_CI_rel_bias_highT['shuffling-{}-{}'.format(setup, rec_length)
        #                          ] = plots.get_CI_median(rel_bias_shuffling_highT)
        median_rel_bias_T_D['bbc-{}-{}'.format(setup, rec_length)
                            ] = np.median(rel_bias_bbc_T_D)
        median_CI_rel_bias_T_D['bbc-{}-{}'.format(setup, rec_length)
                               ] = plots.get_CI_median(rel_bias_bbc_T_D)
        median_rel_bias_T_D['shuffling-{}-{}'.format(setup, rec_length)
                            ] = np.median(rel_bias_shuffling_T_D)
        median_CI_rel_bias_T_D['shuffling-{}-{}'.format(setup, rec_length)
                               ] = plots.get_CI_median(rel_bias_shuffling_T_D)
        mean_rel_bias_T_D['bbc-{}-{}'.format(setup, rec_length)
                          ] = np.mean(rel_bias_bbc_T_D)
        mean_CI_rel_bias_T_D['bbc-{}-{}'.format(setup, rec_length)
                             ] = plots.get_CI_mean(rel_bias_bbc_T_D)
        mean_rel_bias_T_D['shuffling-{}-{}'.format(setup, rec_length)
                          ] = np.mean(rel_bias_shuffling_T_D)
        mean_CI_rel_bias_T_D['shuffling-{}-{}'.format(setup, rec_length)
                             ] = plots.get_CI_mean(rel_bias_shuffling_T_D)

        # mean_rel_bias['shuffling-{}-{}'.format(setup, rec_length)] = np.median(
        #     rel_bias_shuffling)
        # std_rel_bias['bbc-{}-{}'.format(setup, rec_length)
        #              ] = np.std(rel_bias_bbc)
        # std_rel_bias['shuffling-{}-{}'.format(setup, rec_length)
        #              ] = np.std(rel_bias_shuffling)

# for regularization in ['bbc', 'shuffling']:
#     for rec_length in rec_lengths:
#         print(rec_length + ' no CV: ',  median_rel_bias_T_D['{}-full_noCV-{}'.format(regularization, rec_length)], median_CI_rel_bias_T_D['{}-full_noCV-{}'.format(regularization, rec_length)],
#               'with CV: ', median_rel_bias_T_D['{}-full-{}'.format(regularization, rec_length)], median_CI_rel_bias_T_D['{}-full-{}'.format(regularization, rec_length)])
"""Plotting"""
fig, (axes) = plt.subplots(2, 2, figsize=(10, 6))
# fig.set_size_inches(4, 3)
for j, setup in enumerate(setups):
    for k, regularization in enumerate(['bbc', 'shuffling']):
        ax = axes[k][j]

        # plot settings
        # ax.set_xlabel(r'past range $T$ [sec]')
        # ax.set_xscale('log')
        # ax.set_xlim((0.01, 3.5))
        # # ax.set_xticks(np.array([1, 10, 50]))
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.spines['bottom'].set_bounds(0.01, 3)
        # # ax.set_xlabel(r'memory depth $T_m$ (sec)')

        x = rec_length_values
        labels = ['1', '3', '5', '10', '20', '45', '90']

        ax.set_xscale('log')
        ax.set_xlim((50, 22500))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.spines['bottom'].set_bounds(50, 5500)
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        # plt.xticks(x, labels, rotation='horizontal')
        # ax.set_xlabel(r'recording time $T_{rec}$ (min)')

        ##### y-axis ####
        # ax.set_ylabel(r'$M$')
        ax.set_ylim((-5, 15))
        # ax.set_yticks([0.0, 0.08, 0.16])
        # ax.spines['left'].set_bounds(.0, 0.16)

        ##### Unset Borders #####
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        # only plot labels and legend for left-hand side
        if setup == 'full':
            # ax.set_xlabel(r'past range $T$ [sec]')
            ax.set_ylabel(r'relative bias for $R(\hat{T}_D)$ [\%]')
        if regularization == 'shuffling':
            ax.set_xlabel(r'recording time $T_{\mathrm{rec}}$ [min]')

        if setup == 'full':
            ax.set_title(
                '{}, embedding-optimized estimate'.format(regularization))
        else:
            ax.set_title('{}, cross-validation'.format(regularization))
        ax.plot(rec_length_values, np.zeros(len(rec_lengths)), color='0')

        # legend only for no cross-validation
        for i, rec_length in enumerate(rec_lengths):
            median_rel_bias_T_D_val = median_rel_bias_T_D['{}-{}-{}'.format(
                regularization, setup, rec_length)]
            median_CI_rel_bias_T_D_val = median_CI_rel_bias_T_D['{}-{}-{}'.format(
                regularization, setup, rec_length)]
            median_CI_lo = median_CI_rel_bias_T_D_val[0] - \
                median_rel_bias_T_D_val
            median_CI_hi = median_CI_rel_bias_T_D_val[1] - \
                median_rel_bias_T_D_val
            mean_rel_bias_T_D_val = mean_rel_bias_T_D['{}-{}-{}'.format(
                regularization, setup, rec_length)]
            mean_CI_rel_bias_T_D_val = mean_CI_rel_bias_T_D['{}-{}-{}'.format(
                regularization, setup, rec_length)]
            mean_CI_lo = mean_CI_rel_bias_T_D_val[0] - mean_rel_bias_T_D_val
            mean_CI_hi = mean_CI_rel_bias_T_D_val[1] - mean_rel_bias_T_D_val
            # mean_rel_bias_arr = mean_rel_bias['{}-{}-{}'.format(
            #     regularization, setup, rec_length)]
            # std_rel_bias_arr = std_rel_bias['{}-{}-{}'.format(
            #     regularization, setup, rec_length)]
            # ax.plot(T, mean_rel_bias_arr,
            #         linewidth=1.,
            #         color=rec_length_colors['{}-{}'.format(
            #             regularization, rec_length)],
            #         label=rec_length,
            #         zorder=4)
            ax.errorbar(x=[rec_length_values[i]], y=[mean_rel_bias_T_D_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                        color=rec_length_colors['{}-45min'.format(
                            regularization)], marker='d', markersize=5.)
            # ax.fill_between(T, mean_rel_bias_arr - std_rel_bias_arr,
            #                 mean_rel_bias_arr + std_rel_bias_arr,
            #                 facecolor=rec_length_colors['{}-{}'.format(
            #                     regularization, rec_length)],
            #                 alpha=0.4)
        if setup == 'full':
            ax.legend(loc=(.6, .3), frameon=False)
    # sns.palplot(sns.color_palette("RdBu_r", 15))  #visualize the color palette
    # ax.set_color_cycle(sns.color_palette("coolwarm_r",num_lines)) setting it
    # as color cycle to do automatised color assignment

##########################################
########## Simulated Conventional ########
##########################################

# ax.text(0.012, M_max + 0.02 * M_max, r'$\hat{R}_{tot}$')
# ax.text(T_D_shuffling + 0.15 * Tm_eff, .101, r'$\hat{T}_D$')

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# fig.savefig("Subsampling_fixed_Cat.pdf")
plt.savefig('{}/Rtot_bias_vs_Trec_comparison.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
# plt.savefig('../Mopt_vs_Tm_model_comparison.png',
#             format="png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
