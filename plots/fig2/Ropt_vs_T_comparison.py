
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
fig = 'fig2'
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)

if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl

sample_index = 0
rec_length = '90min'
setup = 'full'

"""Settings"""
# Load settings from yaml file
with open('{}/settings/Simulation_{}.yaml'.format(ESTIMATOR_DIR, setup), 'r') as analysis_settings_file:
    analysis_settings = yaml.load(
        analysis_settings_file, Loader=yaml.BaseLoader)

ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
analysis_num_str = glm.load_embedding_parameters(
    rec_length, sample_index, analysis_settings)[2]

"""Load data """

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

# Embedding optimized estimates and confidence intervals
hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
    ANALYSIS_DIR, analysis_num_str)
hisdep_pd = pd.read_csv(hisdep_csv_file_name)

T = np.array(hisdep_pd['#T'])

R_bbc = np.array(hisdep_pd['max_R_bbc'])
R_bbc_CI_lo = np.array(hisdep_pd['max_R_bbc_CI_lo'])
R_bbc_CI_hi = np.array(hisdep_pd['max_R_bbc_CI_hi'])

R_shuffling = np.array(hisdep_pd['max_R_shuffling'])
R_shuffling_CI_lo = np.array(hisdep_pd['max_R_shuffling_CI_lo'])
R_shuffling_CI_hi = np.array(hisdep_pd['max_R_shuffling_CI_hi'])

glm_bbc_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_bbc.csv'.format(
    ANALYSIS_DIR, analysis_num_str)
glm_bbc_pd = pd.read_csv(glm_bbc_csv_file_name)
R_glm_bbc = np.array(glm_bbc_pd['R_GLM'])

glm_shuffling_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_shuffling.csv'.format(
    ANALYSIS_DIR, analysis_num_str)
glm_shuffling_pd = pd.read_csv(glm_shuffling_csv_file_name)
R_glm_shuffling = np.array(glm_shuffling_pd['R_GLM'])

# Temporal depth and total history dependence
statistics_csv_file_name = '{}/ANALYSIS{}/statistics.csv'.format(
    ANALYSIS_DIR, analysis_num_str)
statistics_pd = pd.read_csv(statistics_csv_file_name)

R_tot_bbc = statistics_pd['R_tot_bbc'][0]
T_D_bbc = statistics_pd['T_D_bbc'][0]

R_tot_shuffling = statistics_pd['R_tot_shuffling'][0]
T_D_shuffling = statistics_pd['T_D_shuffling'][0]

bbc_tolerance = statistics_pd['bbc_tolerance'][0]

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
ax.plot([T[0], T[-1]], [R_max, R_max], '--', color='0.5', zorder=1)
ax.plot(T_old, R_GLM_BIC, color='.5', zorder=3)

# GLM for same embeddings as comparison
# ax.plot(T, R_glm_bbc, '-.', color='.4', alpha=0.8,
#         zorder=3, label='true $R(T)$ (BBC)')  # , label='Model'
# ax.plot(T, R_glm_shuffling, ':', color='.4',
#         lw=1.8, alpha=0.8, zorder=2, label=r'true $R(T)$ (Shuffling)')

# Embedding optimized estimates and confidence intervals
ax.plot(T, R_bbc, linewidth=1.2,  color=main_red, zorder=4)
ax.fill_between(T, R_bbc_CI_lo, R_bbc_CI_hi, facecolor=main_red, alpha=0.3)
ax.plot(T, R_shuffling, linewidth=1.2, color=main_blue, zorder=3)
ax.fill_between(T, R_shuffling_CI_lo, R_shuffling_CI_hi,
                facecolor=main_blue, alpha=0.3)

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
# fig.savefig("Subsampling_fixed_Cat.pdf")
# plt.savefig('{}/Ropt_vs_T_comparison.pdf'.format(PLOTTING_DIR),
#             format="pdf", bbox_inches='tight')
# plt.savefig('../Mopt_vs_Tm_model_comparison.png',
#             format="png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
