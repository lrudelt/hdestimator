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
sys.path.append('../../Scripts/Functions')
# import plotutils

"""Parameters and Settings"""
recorded_system = 'EC'
rec_length = '90min'

# ESTIMATOR_DIR = '{}/../..'.format(dirname(realpath(__file__)))
ESTIMATOR_DIR = '/home/lucas/research/projects/history_dependence/hdestimator'
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))
if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl

# fig = dirname(realpath(__file__)).split("/")[-1]
fig = 'fig3'
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)

  # Example neuron for EC
  neuron_index = 10  # Neuron 38
   # neuron_index = 14 # Neuron 44? This one could also work

   def load_analysis_results_full(recorded_system, setup, ESTIMATOR_DIR):
        with open('{}/settings/{}_{}.yaml'.format(ESTIMATOR_DIR, recorded_system, setup), 'r') as analysis_settings_file:
            analysis_settings = yaml.load(
                analysis_settings_file, Loader=yaml.BaseLoader)
        ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
        # Get Rtot, TD and analysis_num from statistics.scv
        statistics_merged_csv_file_name = '{}/statistics_merged.csv'.format(
            ANALYSIS_DIR)
        statistics_pd = pd.read_csv(statistics_merged_csv_file_name)

        index = np.where(statistics_pd['label'] == rec_length +
                         "-" + setup + "-" + str(neuron_index))[0][0]
        analysis_num_str = str(statistics_pd['#analysis_num'][index])
        for i in range(4 - len(analysis_num_str)):
            analysis_num_str = '0' + analysis_num_str
        R_tot_bbc = statistics_pd['R_tot_bbc'][index]
        T_D_bbc = statistics_pd['T_D_bbc'][index]
        R_tot_shuffling = statistics_pd['R_tot_shuffling'][index]
        T_D_shuffling = statistics_pd['T_D_shuffling'][index]
        # Get T, R(T) plus confidence intervals
        hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
            ANALYSIS_DIR, analysis_num_str)
        hisdep_pd = pd.read_csv(hisdep_csv_file_name)
        R_bbc = np.array(hisdep_pd['max_R_bbc'])
        R_bbc_CI_lo = np.array(hisdep_pd['max_R_bbc_CI_lo'])
        R_bbc_CI_hi = np.array(hisdep_pd['max_R_bbc_CI_hi'])
        R_shuffling = np.array(hisdep_pd['max_R_shuffling'])
        R_shuffling_CI_lo = np.array(hisdep_pd['max_R_shuffling_CI_lo'])
        R_shuffling_CI_hi = np.array(hisdep_pd['max_R_shuffling_CI_hi'])
        T = np.array(hisdep_pd['#T'])
        return ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, R_tot_shuffling, T_D_shuffling, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi

    def load_analysis_results_shuffling(recorded_system, setup, ESTIMATOR_DIR):
        with open('{}/settings/{}_{}.yaml'.format(ESTIMATOR_DIR, recorded_system, setup), 'r') as analysis_settings_file:
            analysis_settings = yaml.load(
                analysis_settings_file, Loader=yaml.BaseLoader)
        ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
        # Get Rtot, TD and analysis_num from statistics.scv
        statistics_merged_csv_file_name = '{}/statistics_merged.csv'.format(
            ANALYSIS_DIR)
        statistics_pd = pd.read_csv(statistics_merged_csv_file_name)

        index = np.where(statistics_pd['label'] == rec_length +
                         "-" + setup + "-" + str(neuron_index))[0][0]
        analysis_num_str = str(statistics_pd['#analysis_num'][index])
        for i in range(4 - len(analysis_num_str)):
            analysis_num_str = '0' + analysis_num_str
        R_tot_shuffling = statistics_pd['R_tot_shuffling'][index]
        T_D_shuffling = statistics_pd['T_D_shuffling'][index]
        # Get T, R(T) plus confidence intervals
        hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
            ANALYSIS_DIR, analysis_num_str)
        hisdep_pd = pd.read_csv(hisdep_csv_file_name)
        R_shuffling = np.array(hisdep_pd['max_R_shuffling'])
        R_shuffling_CI_lo = np.array(hisdep_pd['max_R_shuffling_CI_lo'])
        R_shuffling_CI_hi = np.array(hisdep_pd['max_R_shuffling_CI_hi'])
        T = np.array(hisdep_pd['#T'])
        return R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi

    statistics_pd['bs_CI_percentile_lo'][index]
    statistics_pd.columns

    """Load data full"""
    setup = 'full_noCV'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, R_tot_shuffling, T_D_shuffling, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = load_analysis_results_full(
        recorded_system, setup, ESTIMATOR_DIR)

    # Get R_tot_glm for T_D
    glm_csv_file_name = '{}/ANALYSIS{}/glm_estimates_BIC.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    glm_pd = pd.read_csv(glm_csv_file_name)
    index = np.argmin(glm_pd['BIC'])
    R_tot_glm = glm_pd['R_GLM'][index]

    """Load data five bins"""
    setup = 'fast_noCV'
    R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = load_analysis_results_shuffling(
        recorded_system, setup, ESTIMATOR_DIR)

    """Load data onebins"""
    setup = 'onebin_noCV'
    R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = load_analysis_results_shuffling(
        recorded_system, setup, ESTIMATOR_DIR)

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

    fig, ((ax)) = plt.subplots(1, 1, figsize=(6.5, 2.8))

    ax.set_xscale('log')
    ax.set_xlim((0.01, 10.))
    # ax.set_xticks(np.array([1, 10, 50]))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.spines['bottom'].set_bounds(0.01, 10.)
    ax.set_xlabel(r'past range $T$ [sec]')
    ax.set_ylabel(r'history dependence $R(T)$')
    ##### y-axis ####

    max_val = np.amax(R_tot_shuffling)
    if max_val > 0.2:
        if max_val > 0.3:
            if max_val > 0.405:
                ax.set_ylim((0.1, .51))
                ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
                ax.spines['left'].set_bounds(.1, 0.5)
            else:
                ax.set_ylim((0.1, .4))
                ax.set_yticks([0.1, 0.2, 0.3, 0.4])
                ax.spines['left'].set_bounds(.1, 0.4)
        else:
            ax.set_ylim((0.0, .3))
            ax.set_yticks([0.0, 0.1, 0.2, 0.3])
            ax.spines['left'].set_bounds(.0, 0.3)
    else:
        ax.set_ylim((0.0, .2))
        ax.set_yticks([0.0, 0.1, 0.2])
        ax.spines['left'].set_bounds(.0, 0.2)

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)

    ax.plot(T, R_bbc, linewidth=1.2,  color=main_red,
            label='BBC', zorder=4)
    ax.fill_between(T, R_bbc_CI_lo, R_bbc_CI_hi,
                    facecolor=main_red, alpha=0.3)
    ax.plot(T, R_shuffling, linewidth=1.2,  color=main_blue,
            label='Shuffling', zorder=4)
    ax.fill_between(T, R_shuffling_CI_lo, R_shuffling_CI_hi,
                    facecolor=main_blue, alpha=0.3)
    ax.plot(T, R_fivebins, linewidth=1.2,  color=green,
            label='max five bins', zorder=4)
    ax.fill_between(T, R_fivebins_CI_lo, R_fivebins_CI_hi,
                    facecolor=green, alpha=0.3)
    ax.plot(T, R_onebin, linewidth=1.2,  color='y',
            label='one bin', zorder=4)
    ax.fill_between(T, R_onebin_CI_lo, R_onebin_CI_hi,
                    facecolor='y', alpha=0.3)
    R_max = np.amax(R_bbc)
    R_tot_bbc
    R_eff = 0.98 * R_max
    T[R_shuffling > R_eff]
    # R_tot_bbc
    # R_bbc
    # Plot R_tot and T_D for BBC
    ax.axvline(x=T_D_bbc, ymax=0.9, color=main_red,
               linewidth=0.5, linestyle='--')
    ax.axhline(y=R_tot_bbc, xmax=.67, color=main_red,
               linewidth=0.5, linestyle='--')
    ax.plot([0.01], [R_tot_bbc], marker='s', markersize=1.5, color=main_red,
            zorder=8)
    ax.plot([T_D_bbc], [0.1], marker='s', markersize=1.5, color=main_red,
            zorder=8)
    ax.text(0.012, R_tot_bbc + 0.02 * R_tot_bbc, r'$\hat{R}_{\mathrm{tot}}$')
    ax.text(T_D_bbc + 0.15 * T_D_bbc, .105, r'$\hat{T}_D$')
    # Plot R_tot and T_D for Shuffling
    ax.axvline(x=T_D_shuffling, ymax=0.9, color=main_blue,
               linewidth=0.5, linestyle='--')
    ax.axhline(y=R_tot_shuffling, xmax=.67, color=main_blue,
               linewidth=0.5, linestyle='--')
    ax.plot([0.01], [R_tot_shuffling], marker='s', markersize=1.5, color=main_blue,
            zorder=8)
    ax.plot([T_D_shuffling], [0.1], marker='s', markersize=1.5, color=main_blue,
            zorder=8)
    # Plot R_tot and T_D for max five bins
    ax.axvline(x=T_D_fivebins, ymax=0.9, color=green,
               linewidth=0.5, linestyle='--')
    ax.axhline(y=R_tot_fivebins, xmax=.67, color=green,
               linewidth=0.5, linestyle='--')
    ax.plot([0.01], [R_tot_fivebins], marker='s', markersize=1.5, color=green,
            zorder=8)
    ax.plot([T_D_fivebins], [0.1], marker='s', markersize=1.5, color=green,
            zorder=8)
    # Plot R_tot and T_D for one bin
    ax.axvline(x=T_D_onebin, ymax=0.9, color='y',
               linewidth=0.5, linestyle='--')
    ax.axhline(y=R_tot_onebin, xmax=.67, color='y',
               linewidth=0.5, linestyle='--')
    ax.plot([0.01], [R_tot_onebin], marker='s', markersize=1.5, color='y',
            zorder=8)
    ax.plot([T_D_onebin], [0.1], marker='s', markersize=1.5, color='y',
            zorder=8)

    # Plot R_tot_glm
    ax.plot([T_D_bbc], [R_tot_glm], 's', color=violet, label='GLM')

    ax.legend(loc=(1.15, 0.05), frameon=False)

    fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.savefig('{}/Ropt_vs_T_EC.pdf'.format(PLOTTING_DIR),
                format="pdf", bbox_inches='tight')
    # plt.savefig('../../Paper_Figures/fig3_experimentalBenchmarks/Ropt_vs_Tp_%s_EC.png' % (rec_length),
    #             format="png", dpi=400, bbox_inches='tight')

    plt.show()
    plt.close()
