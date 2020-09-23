
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
setups = np.array([['full_bbc','full_bbc_withCV'], ['full_shuffling','full_shuffling_withCV']])
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
mean_rel_deviation_R_tot = {}
mean_CI_rel_deviation_R_tot = {}

for setup in setups.flatten():
    # Load settings from yaml file
    with open('{}/settings/Simulation_{}.yaml'.format(ESTIMATOR_DIR, setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)
    ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    T = np.array(analysis_settings['embedding_past_range_set']).astype(float)
    regularization_method = setup.split("_")[1]
    for rec_length in rec_lengths:
        rel_deviation_R_tot = []
        number_samples = 30
        if rec_length == '45min':
            number_samples = 10
        if rec_length == '90min':
            number_samples = 10
        for sample_index in np.arange(1, number_samples):
            analysis_num_str = glm.load_embedding_parameters(
                rec_length, sample_index, analysis_settings)[1]
            # Embedding optimized estimates and confidence intervals
            hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
                ANALYSIS_DIR, analysis_num_str)
            hisdep_pd = pd.read_csv(hisdep_csv_file_name)
            R = np.array(hisdep_pd['max_R_{}'.format(regularization_method)])
            R_CI_lo = np.array(hisdep_pd['max_R_{}_CI_lo'.format(regularization_method)])
            T = np.array(hisdep_pd['#T'])
            R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)

            glm_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_{}.csv'.format(
                ANALYSIS_DIR, analysis_num_str, regularization_method)
            glm_pd = pd.read_csv(glm_csv_file_name)
            T_glm = np.array(glm_pd['T'])
            R_glm = np.array(glm_pd['R_GLM'])
            # Make sure that you only average R_GLM over the right T
            if T_glm[0]>T[T_D_index]:
                print(setup, rec_length, sample_index)
            else:
                T_D_index_glm = np.where(T_glm == T[T_D_index])[0][0]
                max_valid_index_glm = np.where(T_glm == T[max_valid_index-1])[0][0]+1
                R_tot_glm = np.mean(R_glm[T_D_index_glm:max_valid_index_glm])
                rel_deviation_R_tot += [100* (R_tot-R_tot_glm)/R_tot_glm]

        mean_rel_deviation_R_tot['{}-{}'.format(setup, rec_length)
                          ] = np.mean(rel_deviation_R_tot)
        mean_CI_rel_deviation_R_tot['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(rel_deviation_R_tot)



"""Plotting"""
fig, (axes) = plt.subplots(2, 2, figsize=(10, 6))
# fig.set_size_inches(4, 3)
for j in range(2):
    for k in range(2):
        setup = setups[k][j]
        ax = axes[k][j]
        regularization_method = setup.split("_")[1]

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

        ##### y-axis ####
        ax.set_ylim((-10, 10))

        ##### Unset Borders #####
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        # only plot labels and legend for left-hand side
        if not setup.split('_')[-1] == 'withCV':
            # ax.set_xlabel(r'past range $T$ [sec]')
            ax.set_ylabel(r'relative bias for $\hat{R}_{\mathrm{tot}}$ [\%]')
        if regularization_method == 'shuffling':
            ax.set_xlabel(r'recording time $T_{\mathrm{rec}}$ [min]')

        if not setup.split('_')[-1] == 'withCV':
            ax.set_title(
                '{}, embedding-optimized estimate'.format(regularization_method))
        else:
            ax.set_title('{}, cross-validation'.format(regularization_method))
        ax.plot(rec_length_values, np.zeros(len(rec_lengths)), color='0')

        for i, rec_length in enumerate(rec_lengths):
            mean_rel_deviation_R_tot_val = mean_rel_deviation_R_tot['{}-{}'.format(
                setup, rec_length)]
            mean_CI_rel_deviation_R_tot_val = mean_CI_rel_deviation_R_tot['{}-{}'.format(
                setup, rec_length)]
            mean_CI_lo = mean_CI_rel_deviation_R_tot_val[0] - mean_rel_deviation_R_tot_val
            mean_CI_hi = mean_CI_rel_deviation_R_tot_val[1] - mean_rel_deviation_R_tot_val

            ax.errorbar(x=[rec_length_values[i]], y=[mean_rel_deviation_R_tot_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                        color=rec_length_colors['{}-45min'.format(
                            regularization_method)], marker='d', markersize=5.)


# ax.text(0.012, M_max + 0.02 * M_max, r'$\hat{R}_{tot}$')

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('{}/Rtot_bias_vs_Trec_comparison.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
