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

"""Parameters and Settings"""
recorded_system = 'EC'
rec_length = '90min'
number_valid_neurons = 28

# ESTIMATOR_DIR = '{}/../..'.format(dirname(realpath(__file__)))
ESTIMATOR_DIR = '/home/lucas/research/projects/history_dependence/hdestimator'
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))
if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots

# fig = dirname(realpath(__file__)).split("/")[-1]
fig = 'fig3'
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)

R_bbc_uniform_list = []
R_fivebins_uniform_list = []
"""Load data"""
for neuron_index in range(number_valid_neurons):
    setup = 'full_bbc'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'bbc')

    setup = 'full_bbc_uniform'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc_uniform, T_D_bbc_uniform, T, R_bbc_uniform, R_bbc_uniform_CI_lo, R_bbc_uniform_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'bbc')
    R_bbc_uniform = R_bbc_uniform/R_bbc
    R_bbc_uniform_list += [R_bbc_uniform]

    setup = 'fivebins'
    R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')

    setup = 'fivebins_uniform'
    R_tot_fivebins_uniform, T_D_fivebins_uniform, T, R_fivebins_uniform, R_fivebins_uniform_CI_lo, R_fivebins_uniform_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    R_fivebins_uniform = R_fivebins_uniform/R_bbc
    R_fivebins_uniform_list += [R_fivebins_uniform]

R_bbc_uniform_median = np.sort(np.transpose(R_bbc_uniform_list), axis=1)[
    :, int(number_valid_neurons / 2)]
R_bbc_uniform_loPC = np.sort(np.transpose(R_bbc_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.05)]
R_bbc_uniform_hiPC = np.sort(np.transpose(R_bbc_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.95)]
R_fivebins_uniform_median = np.sort(np.transpose(R_fivebins_uniform_list), axis=1)[
    :, int(number_valid_neurons / 2)]
R_fivebins_uniform_loPC = np.sort(np.transpose(R_fivebins_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.05)]
R_fivebins_uniform_hiPC = np.sort(np.transpose(R_fivebins_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.95)]


"""Plotting"""
# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]


fig, ((ax)) = plt.subplots(1, 1, figsize=(3.5, 2.8))

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6


ax.set_xscale('log')
ax.set_xlim((0.01, 5.))
ax.set_xticks(np.array([0.01, 0.1, 1]))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.spines['bottom'].set_bounds(0.01, 5)
ax.set_xlabel(r'past range  $T$ [sec]')

##### y-axis ####
ax.set_ylabel(
    r'\begin{center}history dependence $R(T)$ \\ relative to exponential\end{center}')
# ax.set_ylim((0.0, 1.))
# ax.set_yticks([0.0, 0.5, 1.0])
# ax.spines['left'].set_bounds(.0, 1.)


##### Unset Borders #####
ax.spines['top'].set_bounds(0, 0)
ax.spines['right'].set_bounds(0, 0)

ax.plot(T, R_bbc_uniform_median,
        linewidth=2,  color=soft_red, zorder=1)
ax.fill_between(T, R_bbc_uniform_loPC, R_bbc_uniform_hiPC,
                facecolor=soft_red, alpha=0.3)

ax.plot(T, R_fivebins_uniform_median,
        linewidth=2, alpha=0.5, color='g', zorder=1)
ax.fill_between(T, R_fivebins_uniform_loPC, R_fivebins_uniform_hiPC,
                facecolor='g', alpha=0.3)
ax.plot(T, np.zeros(len(T))+1,
        linewidth=2, color=main_red, zorder=2)
ax.plot(T, np.zeros(len(T))+1,
        linewidth=2.5, color='g', zorder=1)
ax.text(0.1, 0.6, r'\begin{center}uniform \\($\kappa = 0$)\end{center}',
        color='0.0', ha='left', va='bottom')
ax.text(0.01, 1.02, r'exponential ($\kappa$ optimized)',
        color='0.0', ha='left', va='bottom')

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# fig.savefig("Subsampling_fixed_Cat.pdf")

plt.savefig('{}/Ropt_vs_T_uniexp_relative.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
