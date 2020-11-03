
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

# fig = dirname(realpath(__file__)).split("/")[-1]
fig = 'fig3'
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)

if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots

recorded_system = 'Simulation'
rec_length = '90min'
sample_index = 0

"""Load data """
# load estimate of ground truth
R_tot_true = np.load('{}/analysis_data/R_tot_simulation.npy'.format(ESTIMATOR_DIR))
T_true, R_true = plots.load_analysis_results_glm_Simulation(ESTIMATOR_DIR)

# Load settings from yaml file
setup = 'full_bbc'

ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
    recorded_system, rec_length, sample_index, setup, ESTIMATOR_DIR, regularization_method='bbc')

R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T, R_bbc, R_bbc_CI_lo)

glm_bbc_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_bbc.csv'.format(
    ANALYSIS_DIR, analysis_num_str)
glm_bbc_pd = pd.read_csv(glm_bbc_csv_file_name)
R_glm_bbc = np.array(glm_bbc_pd['R_GLM'])

setup = 'full_shuffling'

ANALYSIS_DIR, analysis_num_str, R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
    recorded_system, rec_length, sample_index, setup, ESTIMATOR_DIR, regularization_method='shuffling')

R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T, R_shuffling, R_shuffling_CI_lo)

glm_shuffling_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_shuffling.csv'.format(
    ANALYSIS_DIR, analysis_num_str)
glm_shuffling_pd = pd.read_csv(glm_shuffling_csv_file_name)
R_glm_shuffling = np.array(glm_shuffling_pd['R_GLM'])

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ((ax)) = plt.subplots(1, 1, figsize=(3.5, 3))

# fig.set_size_inches(4, 3)

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
soft_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
# sns.palplot(sns.color_palette("RdBu_r", 15))  #visualize the color palette
# ax.set_color_cycle(sns.color_palette("coolwarm_r",num_lines)) setting it
# as color cycle to do automatised color assignment

##########################################
########## Simulated Conventional ########
##########################################

ax.set_xscale('log')
ax.set_xlim((0.1, 3.5))
# ax.set_xticks(np.array([1, 10, 50]))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.spines['bottom'].set_bounds(0.1, 3)
# ax.set_xlabel(r'memory depth $T_m$ (sec)')

##### y-axis ####
# ax.set_ylabel(r'$M$')
ax.set_ylim((0.1, .14))
ax.set_yticks([0.1, 0.12, 0.14])
ax.spines['left'].set_bounds(.1, 0.14)

##### Unset Borders #####
ax.spines['top'].set_bounds(0, 0)
ax.spines['right'].set_bounds(0, 0)

# GLM true max and R vs T
ax.plot([T[0], T[-1]], [R_tot_true, R_tot_true], '--', color='0.5', zorder=1)
ax.plot(T_true, R_true, color='.5', zorder=4)

# GLM for same embeddings as comparison
ax.plot(T, R_glm_bbc, '-.', color='.4', alpha=0.8,
        zorder=3, label='true $R(T)$ (BBC)')  # , label='Model'
ax.plot(T, R_glm_shuffling, ':', color='.4',
        lw=1.8, alpha=0.8, zorder=2, label=r'true $R(T)$ (Shuffling)')

# Embedding optimized estimates and confidence intervals
ax.plot(T, R_bbc, linewidth=1.2,  color=main_red, zorder=4)
ax.fill_between(T, R_bbc_CI_lo, R_bbc_CI_hi, facecolor=main_red, alpha=0.3)
ax.plot(T, R_shuffling, linewidth=1.2, color=main_blue, zorder=3)
ax.fill_between(T, R_shuffling_CI_lo, R_shuffling_CI_hi,
                facecolor=main_blue, alpha=0.3)

ax.plot(T[T_D_index_bbc:max_valid_index_bbc], np.zeros(max_valid_index_bbc-T_D_index_bbc)+R_tot_bbc, color=main_red, linestyle='--')
ax.plot(T[T_D_index_shuffling:max_valid_index_shuffling], np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_shuffling, color=main_blue, linestyle='--')


# Rtot and Tdepth bbc

ax.axvline(x=T_D_bbc, ymax=0.7, color=main_red,
           linewidth=0.5, linestyle='--')
ax.axhline(y=R_tot_bbc, xmax=.5, color=main_red,
           linewidth=0.5, linestyle='--')
ax.plot([0.1], [R_tot_bbc], marker='d', markersize=3, color=main_red,
        zorder=8)
ax.plot([T_D_bbc], [0.1], marker='d', markersize=3, color=main_red,
        zorder=8)
ax.plot([T_D_bbc], [R_tot_bbc], marker='x', markersize=6, color=main_red,
        zorder=8)
# ax.text(0.012, M_max + 0.02 * M_max, r'$\hat{R}_{tot}$')
ax.text(T_D_bbc + 0.15 * T_D_bbc, .101, r'$\hat{T}_D$')

# Rtot and Tdepth Shuffling

ax.axvline(x=T_D_shuffling, ymax=0.6, color=main_blue,
           linewidth=0.5, linestyle='--')
ax.axhline(y=R_tot_shuffling, xmax=.45, color=main_blue,
           linewidth=0.5, linestyle='--')
ax.plot([0.1], [R_tot_shuffling], marker='d', markersize=3, color=main_blue,
        zorder=8)
ax.plot([T_D_shuffling], [0.1], marker='d', markersize=3, color=main_blue,
        zorder=8)
ax.plot([T_D_shuffling], [R_tot_shuffling], marker='x', markersize=6, color=main_blue,
        zorder=8)
# ax.text(0.012, M_max + 0.02 * M_max, r'$\hat{R}_{tot}$')
# ax.text(T_D_shuffling + 0.15 * Tm_eff, .101, r'$\hat{T}_D$')

ax.legend(loc=(.05, .83), frameon=False)

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.savefig('{}/Ropt_vs_T_comparison.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
