
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


def median_relative_mean_R_tot_and_T_D(recorded_system, setup, N_neurons, rec_lengths, rec_lengths_Nsamples, ESTIMATOR_DIR):
    if recorded_system == 'EC':
        dataDir = '/data.nst/lucas/history_dependence/paper/EC_data/'
    if recorded_system == 'Retina':
        dataDir = '/data.nst/lucas/history_dependence/paper/retina_data/'
    if recorded_system == 'Culture':
        dataDir = '/data.nst/lucas/history_dependence/paper/culture_data/'
    validNeurons = np.load(
        '{}validNeurons.npy'.format(dataDir)).astype(int)
    R_tot_relative_mean = {}
    T_D_relative_mean = {}
    np.random.seed(41)
    neuron_selection = np.random.choice(len(validNeurons), N_neurons,  replace=False)
    for rec_length in rec_lengths:
        # arrays containing R_tot and mean T_D for different neurons
        R_tot_mean_arr = []
        T_D_mean_arr = []
        N_samples = rec_lengths_Nsamples[rec_length]
        for j in range(N_neurons):
            neuron_index = neuron_selection[j]
            R_tot_arr = []
            T_D_arr = []
            for sample_index in range(N_samples):
                # Get run index
                run_index = j * N_samples + sample_index
                """Load data five bins"""
                if not rec_length == '90min':
                    setup_subsampled = '{}_subsampled'.format(setup)
                else:
                    run_index = neuron_index
                    setup_subsampled = setup
                if setup == 'full_bbc':
                    analysis_results = plots.load_analysis_results(
                        recorded_system, rec_length, run_index, setup_subsampled, ESTIMATOR_DIR, regularization_method='bbc')
                else:
                    analysis_results = plots.load_analysis_results(
                        recorded_system, rec_length, run_index, setup_subsampled, ESTIMATOR_DIR, regularization_method='shuffling')
                if not analysis_results == None:
                    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = analysis_results
                    # print(analysis_num_str, len(R), rec_length, j, sample_index)
                    # print(R - R_CI_lo)
                    if not len(R) == 0:
                        R_tot_analysis_results = plots.get_R_tot(T, R, R_CI_lo)
                        # if not len(R)<50:
                        #     T_D, R_tot, T_D_index, max_valid_index = plots.get_temporal_depth_and_R_tot(T, R)
                        if not R_tot_analysis_results == None:
                            R_tot, T_D_index, max_valid_index = R_tot_analysis_results
                            R_tot_arr += [R_tot]
                            T_D_arr += [T_D]
                        else:
                            print('CI_fail', recorded_system, setup, rec_length, run_index, neuron_index, sample_index)
                    else:
                        print('no valid embeddings', recorded_system, rec_length,  setup, analysis_num_str)
                else:
                    print('analysis_fail', recorded_system, rec_length, setup, run_index, neuron_index, sample_index)
            R_tot_mean_arr += [np.mean(R_tot_arr)]
            T_D_mean_arr += [np.mean(T_D_arr)]
        R_tot_relative_mean[rec_length] = np.array(R_tot_mean_arr)
        T_D_relative_mean[rec_length] = np.array(T_D_mean_arr)

    median_R_tot_relative_mean = []
    median_CI_R_tot_relative_mean = []
    median_T_D_relative_mean = []
    median_CI_T_D_relative_mean = []
    for rec_length in rec_lengths:
        R_tot_relative_mean_arr = R_tot_relative_mean[rec_length] / R_tot_relative_mean['90min']*100
        T_D_relative_mean_arr = T_D_relative_mean[rec_length] / T_D_relative_mean['90min']*100
        # If no valid embeddings were found for BBC for all samples, the mean is nan so the neuron is not considered in the median operation
        R_tot_relative_mean_arr = R_tot_relative_mean_arr[~np.isnan(R_tot_relative_mean_arr)]
        T_D_relative_mean_arr = T_D_relative_mean_arr[~np.isnan(T_D_relative_mean_arr)]
        # Computing the median and 95% CIs over the 10 neurons
        median_R_tot_relative_mean += [np.median(R_tot_relative_mean_arr)]
        median_CI_R_tot_relative_mean += [plots.get_CI_median(R_tot_relative_mean_arr)]
        median_T_D_relative_mean += [np.median(T_D_relative_mean_arr)]
        median_CI_T_D_relative_mean += [plots.get_CI_median(T_D_relative_mean_arr)]
    return np.array(median_R_tot_relative_mean), np.array(median_CI_R_tot_relative_mean), np.array(median_T_D_relative_mean), np.array(median_CI_T_D_relative_mean)


# fig = dirname(realpath(__file__)).split("/")[-1]
fig = 'supplementaries'
recorded_system = "Retina"
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)
N_neurons = 10

# 1min excluded, because just too little data for neurons with low firing rates
rec_lengths = ['3min', '5min', '10min', '20min', '45min', '90min']
rec_length_values = [180., 300., 600., 1200., 2700., 5400.]
rec_lengths_Nsamples = {'1min': 10, '3min': 10, '5min': 10,
                        '10min': 8, '20min': 4, '45min': 2, '90min': 1}

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6


# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]


"""Load data for all plots"""

"""Plotting"""
fig, (axes) = plt.subplots(3, 2, figsize=(10, 9.5))
# fig.set_size_inches(4, 3)
for i, recorded_system in enumerate(['EC', 'Retina', 'Culture']):
    setup = 'fivebins'
    median_R_tot_relative_mean_fivebins, median_CI_R_tot_relative_mean_fivebins, median_T_D_relative_mean_fivebins, median_CI_T_D_relative_mean_fivebins = median_relative_mean_R_tot_and_T_D(recorded_system, setup, N_neurons, rec_lengths, rec_lengths_Nsamples, ESTIMATOR_DIR)
    # if not recorded_system == "EC":
    setup = 'full_bbc'
    median_R_tot_relative_mean_bbc, median_CI_R_tot_relative_mean_bbc, median_T_D_relative_mean_bbc, median_CI_T_D_relative_mean_bbc = median_relative_mean_R_tot_and_T_D(recorded_system, setup, N_neurons, rec_lengths, rec_lengths_Nsamples, ESTIMATOR_DIR)
    setup = 'full_shuffling'
    median_R_tot_relative_mean_shuffling, median_CI_R_tot_relative_mean_shuffling, median_T_D_relative_mean_shuffling, median_CI_T_D_relative_mean_shuffling = median_relative_mean_R_tot_and_T_D(recorded_system, setup, N_neurons, rec_lengths, rec_lengths_Nsamples, ESTIMATOR_DIR)
    # if not recorded_system == "EC":
    for j in range(2):
        ax = axes[i][j]
        x = rec_length_values
        labels = ['3', '5', '10', '20', '45', '90']

        ax.set_xscale('log')
        ax.set_xlim((170, 22500))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.spines['bottom'].set_bounds(170, 5500)
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r'recording time $T_{\mathrm{rec}}$ [min]')

        ##### y-axis ####

        ##### Unset Borders #####
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        ax.plot(rec_length_values, np.zeros(len(rec_lengths))+100, linestyle='--', color='0')
        # only plot labels and legend for left-hand side
        if j == 0:
            # ax.set_xlabel(r'past range $T$ [sec]')
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
            if i == 0:
                ax.set_ylabel(r'\begin{center}total history dependence $\hat{R}_{\mathrm{tot}}$ \\ relative to $90\,\mathrm{min}$ [\%]\end{center}')
            # ax.text(100, R_tot_true + 0.02 * R_tot_true, r'$\hat{R}_{tot}$')
            # ax.plot(rec_length_values, np.zeros(len(rec_lengths))+R_tot_true, linestyle = '--', color='0')
        else:
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
            if i == 0:
                ax.set_ylabel(r'\begin{center}temporal depth $\hat{T}_D$\\ relative to $90\,\mathrm{min}$ [\%]\end{center}')
            else:
                ax.set_title(r'bla bla', alpha=0.0)

        for k in range(len(rec_lengths)):
            if j == 0:
                median_val_fivebins = median_R_tot_relative_mean_fivebins[k]
                median_CI_val_fivebins = median_CI_R_tot_relative_mean_fivebins[k]
                # if not recorded_system == 'EC':
                median_val_bbc = median_R_tot_relative_mean_bbc[k]
                median_CI_val_bbc = median_CI_R_tot_relative_mean_bbc[k]
                median_val_shuffling = median_R_tot_relative_mean_shuffling[k]
                median_CI_val_shuffling = median_CI_R_tot_relative_mean_shuffling[k]
                # if not recorded_system == 'EC':
            else:

                median_val_fivebins = median_T_D_relative_mean_fivebins[k]
                median_CI_val_fivebins = median_CI_T_D_relative_mean_fivebins[k]
                # if not recorded_system == 'EC':
                median_val_bbc = median_T_D_relative_mean_bbc[k]
                median_CI_val_bbc = median_CI_T_D_relative_mean_bbc[k]
                median_val_shuffling = median_T_D_relative_mean_shuffling[k]
                median_CI_val_shuffling = median_CI_T_D_relative_mean_shuffling[k]
                # if not recorded_system == 'EC':
                # ax.set_ylim((40, 100.5))
            median_CI_hi_fivebins = median_CI_val_fivebins[1] - median_val_fivebins
            median_CI_lo_fivebins = median_CI_val_fivebins[0] - median_val_fivebins
            ax.errorbar(x=rec_length_values[k], y=median_val_fivebins, yerr=[[-median_CI_lo_fivebins], [median_CI_hi_fivebins]],
                        color=green, marker='d', markersize=5., capsize=3.0)
            # if not recorded_system == 'EC':
            median_CI_hi_bbc = median_CI_val_bbc[1] - median_val_bbc
            median_CI_lo_bbc = median_CI_val_bbc[0] - median_val_bbc
            median_CI_hi_shuffling = median_CI_val_shuffling[1] - median_val_shuffling
            median_CI_lo_shuffling = median_CI_val_shuffling[0] - median_val_shuffling
            ax.errorbar(x=rec_length_values[k], y=median_val_bbc, yerr=[[-median_CI_lo_bbc], [median_CI_hi_bbc]],
                        color=main_red, marker='d', markersize=5., capsize=3.0)
            ax.errorbar(x=rec_length_values[k], y=median_val_shuffling, yerr=[[-median_CI_lo_shuffling], [median_CI_hi_shuffling]],
                        color=main_blue, marker='d', markersize=5., capsize=3.0)

            if i+j+k == 0:
                ax.errorbar(x=rec_length_values[k], y=median_val_bbc, yerr=[[-median_CI_lo_bbc], [median_CI_hi_bbc]],
                            color=main_red, marker='d', markersize=5., capsize=3.0, label=r'BBC, $d_{\mathrm{max}}=20$')
                ax.errorbar(x=rec_length_values[k], y=median_val_shuffling, yerr=[[-median_CI_lo_shuffling], [median_CI_hi_shuffling]],
                            color=main_blue, marker='d', markersize=5., capsize=3.0, label=r'Shuffling, $d_{\mathrm{max}}=20$')
                ax.errorbar(x=rec_length_values[k], y=median_val_fivebins, yerr=[[-median_CI_lo_fivebins], [median_CI_hi_fivebins]],
                            color=green, marker='d', markersize=5., capsize=3.0, label=r'Shuffling, $d_{\mathrm{max}}=5$')

        if i+j == 0:
            ax.legend(loc=(0.1, 0.1), frameon=False)
            # if not recorded_system == 'EC':

fig.text(0.5, .99, r'rat dorsal hippocampus (CA1)', ha='center', va='center', fontsize=20)
fig.text(0.5, .66, r'salamander retina', ha='center', va='center', fontsize=20)
fig.text(0.5, .33, r'rat cortical culture', ha='center', va='center', fontsize=20)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
#
plt.savefig('{}/Rtot_vs_Trec_comparison_experiments.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
