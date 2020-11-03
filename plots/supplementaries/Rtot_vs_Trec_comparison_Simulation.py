
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
recorded_system = "Simulation"
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)

old_data_path = '/home/lucas/research/projects/history_dependence/analysis/Data/Simulated/analysis_Data'
R_tot_true = np.loadtxt("%s/M_max_5ms.dat" % old_data_path)
rec_lengths = ['1min', '3min', '5min', '10min', '20min', '45min', '90min']
rec_length_values = [60., 180., 300., 600., 1200., 2700., 5400.]
rec_lengths_colors_bbc = [sns.color_palette("RdBu_r", 15)[8], sns.color_palette("RdBu_r", 15)[9], sns.color_palette("RdBu_r", 15)[10], sns.color_palette(
    "RdBu_r", 15)[11], sns.color_palette("RdBu_r", 15)[12], sns.color_palette("RdBu_r", 15)[13], sns.color_palette("RdBu_r", 15)[14]]
# Only plot first sample

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]

"""Load data for all plots"""
mean_R_tot = {}
mean_CI_R_tot = {}

mean_T_D = {}
mean_CI_T_D = {}

setup = 'full_shuffling'
with open('{}/settings/Simulation_{}.yaml'.format(ESTIMATOR_DIR, setup), 'r') as analysis_settings_file:
    analysis_settings = yaml.load(
        analysis_settings_file, Loader=yaml.BaseLoader)
ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
T = np.array(analysis_settings['embedding_past_range_set']).astype(float)
regularization_method = setup.split("_")[1]
for rec_length in rec_lengths:
    R_tot_arr = []
    T_D_arr = []
    number_samples = 30
    if rec_length == '45min':
        number_samples = 10
    if rec_length == '90min':
        number_samples = 10
    for sample_index in np.arange(1, number_samples):
        ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
            recorded_system, rec_length, sample_index, setup, ESTIMATOR_DIR, regularization_method = regularization_method)
        R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
        R_tot_arr += [R_tot]
        T_D_arr += [T_D]
    mean_R_tot['{}-{}'.format(setup, rec_length)] = np.mean(R_tot_arr)
    mean_CI_R_tot['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(R_tot_arr)
    mean_T_D['{}-{}'.format(setup, rec_length)] = np.mean(T_D_arr)
    mean_CI_T_D['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(T_D_arr)

for rec_length in rec_lengths:
    mean_R_tot['{}-{}'.format(setup, rec_length)] = mean_R_tot['{}-{}'.format(setup, rec_length)]/mean_R_tot['{}-{}'.format(setup, '90min')]*100
    mean_CI_R_tot['{}-{}'.format(setup, rec_length)] = mean_CI_R_tot['{}-{}'.format(setup, rec_length)]/mean_R_tot['{}-{}'.format(setup, '90min')]*100
    mean_T_D['{}-{}'.format(setup, rec_length)] = mean_T_D['{}-{}'.format(setup, rec_length)]/mean_T_D['{}-{}'.format(setup, '90min')]*100
    mean_CI_T_D['{}-{}'.format(setup, rec_length)] = mean_CI_T_D['{}-{}'.format(setup, rec_length)]/mean_T_D['{}-{}'.format(setup, '90min')]*100


setup = 'full_bbc'
with open('{}/settings/Simulation_{}.yaml'.format(ESTIMATOR_DIR, setup), 'r') as analysis_settings_file:
    analysis_settings = yaml.load(
        analysis_settings_file, Loader=yaml.BaseLoader)
ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
regularization_method = setup.split("_")[1]
for rec_length in rec_lengths:
    R_tot_arr = []
    T_D_arr = []
    number_samples = 30
    if rec_length == '45min':
        number_samples = 10
    if rec_length == '90min':
        number_samples = 10
    for sample_index in np.arange(1, number_samples):
        ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
            recorded_system, rec_length, sample_index, setup, ESTIMATOR_DIR, regularization_method = regularization_method)
        R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
        R_tot_arr += [R_tot]
        T_D_arr += [T_D]
    mean_R_tot['{}-{}'.format(setup, rec_length)] = np.mean(R_tot_arr)
    mean_CI_R_tot['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(R_tot_arr)
    mean_T_D['{}-{}'.format(setup, rec_length)] = np.mean(T_D_arr)
    mean_CI_T_D['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(T_D_arr)

for rec_length in rec_lengths:
    mean_R_tot['{}-{}'.format(setup, rec_length)] = mean_R_tot['{}-{}'.format(setup, rec_length)]/mean_R_tot['{}-{}'.format(setup, '90min')]*100
    mean_CI_R_tot['{}-{}'.format(setup, rec_length)] = mean_CI_R_tot['{}-{}'.format(setup, rec_length)]/mean_R_tot['{}-{}'.format(setup, '90min')]*100
    mean_T_D['{}-{}'.format(setup, rec_length)] = mean_T_D['{}-{}'.format(setup, rec_length)]/mean_T_D['{}-{}'.format(setup, '90min')]*100
    mean_CI_T_D['{}-{}'.format(setup, rec_length)] = mean_CI_T_D['{}-{}'.format(setup, rec_length)]/mean_T_D['{}-{}'.format(setup, '90min')]*100

"""Plotting"""
fig, (axes) = plt.subplots(1, 2, figsize=(10, 3.2))
# fig.set_size_inches(4, 3)
for j, ax in enumerate(axes):

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
    ax.set_xlabel(r'recording time $T_{\mathrm{rec}}$ [min]')

    ##### y-axis ####


    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    ax.plot(rec_length_values, np.zeros(len(rec_lengths))+100, linestyle = '--', color='0')
    # only plot labels and legend for left-hand side
    if j == 0:
        # ax.set_xlabel(r'past range $T$ [sec]')
        ax.set_ylabel(r'\begin{center}total history dependence $\hat{R}_{\mathrm{tot}}$ \\ relative to $90\,\mathrm{min}$ [\%]\end{center}')
        # ax.text(100, R_tot_true + 0.02 * R_tot_true, r'$\hat{R}_{tot}$')
        # ax.plot(rec_length_values, np.zeros(len(rec_lengths))+R_tot_true, linestyle = '--', color='0')
    else:
        ax.set_ylabel(r'\begin{center}temporal depth $\hat{T}_D$\\ relative to $90\,\mathrm{min}$ [\%]\end{center}')

    setup = 'full_bbc'
    for i, rec_length in enumerate(rec_lengths):
        if j == 0:
            mean_val = mean_R_tot['{}-{}'.format(
            setup, rec_length)]
            mean_CI_val = mean_CI_R_tot['{}-{}'.format(
            setup, rec_length)]
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
        else:
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
            # ax.set_ylim((40, 100.5))
            mean_val = mean_T_D['{}-{}'.format(
            setup, rec_length)]
            mean_CI_val = mean_CI_T_D['{}-{}'.format(
            setup, rec_length)]
        mean_CI_lo = mean_CI_val[0] - mean_val
        mean_CI_hi = mean_CI_val[1] - mean_val
        if i == 0:
            ax.errorbar(x=[rec_length_values[i]], y=[mean_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                        color=main_red, marker='d', markersize=5.,capsize = 3.0, label = r'BBC, $d_{\mathrm{max}}=20$')
        else:
            ax.errorbar(x=[rec_length_values[i]], y=[mean_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                        color=main_red, marker='d', markersize=5.,capsize = 3.0)

    setup = 'full_shuffling'
    for i, rec_length in enumerate(rec_lengths):
        if j == 0:
            mean_val = mean_R_tot['{}-{}'.format(
            setup, rec_length)]
            mean_CI_val = mean_CI_R_tot['{}-{}'.format(
            setup, rec_length)]
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
        else:
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
            # ax.set_ylim((40, 100.5))
            mean_val = mean_T_D['{}-{}'.format(
            setup, rec_length)]
            mean_CI_val = mean_CI_T_D['{}-{}'.format(
            setup, rec_length)]
        mean_CI_lo = mean_CI_val[0] - mean_val
        mean_CI_hi = mean_CI_val[1] - mean_val
        if i == 0:
            ax.errorbar(x=[rec_length_values[i]], y=[mean_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                        color=main_blue, marker='d', markersize=5., label = r'Shuffling, $d_{\mathrm{max}}=20$')
        else:
            ax.errorbar(x=[rec_length_values[i]], y=[mean_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                        color=main_blue, marker='d', markersize=5.)



    if j == 0:
        ax.legend(loc=(0.1, 0.1), frameon=False)

fig.text(0.5, 1.01, r'simulation', ha='center', va='center', fontsize = 20)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('{}/Rtot_vs_Trec_comparison_simulation.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
