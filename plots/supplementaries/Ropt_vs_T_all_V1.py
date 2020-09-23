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

"""Load data"""
recorded_system = 'V1'
setup = 'fivebins'
rec_length = '40min'

# '2-303': 104 freak
# '2-338' : 52 normal
# '2-357' : 62 bursty

dataDir = '/data.nst/lucas/history_dependence/paper/neuropixel_data/Waksman/'
analysisDataDir = '/data.nst/lucas/history_dependence/neuropixel_data/Waksman/analysis/'
# plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/Neuropixel/'

validNeuronsAreas = np.load('{}validNeuronsAreas.npy'.format(
    dataDir), allow_pickle=True).item()
areaLayers = {'primaryVisualCortex': ['VISp23', 'VISp4', 'VISp5', 'VISp6b', 'VISp6a'], 'rostrolateralArea': [
    'VISrl4', 'VISrl5', 'VISrl6b', 'VISrl6a'], 'primaryMotorCortex': ['MOp5', 'MOp6a', 'MOp23']}

area = 'primaryVisualCortex'
validNeurons = []
for layer in areaLayers[area]:
    for neuron in validNeuronsAreas[layer]:
        validNeurons += [neuron]


"""Plot"""

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '13.0'
matplotlib.rcParams['xtick.labelsize'] = '13'
matplotlib.rcParams['ytick.labelsize'] = '13'
matplotlib.rcParams['legend.fontsize'] = '13'
matplotlib.rcParams['axes.linewidth'] = 0.6
# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

##### x-axis ####


fig, axes = plt.subplots(18, 8, figsize=(14., 19.5))

# Sort neurons, put neurons with max_val > 0.2 and max_val <0.3 in a separate group

normalR = []
highR = []
veryhighR = []
for neuron_index, neuron in enumerate(validNeurons):
    R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    max_val = np.amax(R)
    if max_val > 0.2:
        if max_val >0.3:
            veryhighR += [neuron_index]
        else:
            highR+= [neuron_index]
    else:
        normalR += [neuron_index]

for k, neuron_index in enumerate(np.append(np.append(highR, normalR),veryhighR)):
    R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
    R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
    # Only plot the confidence interval at Rmax for consistency
    max_index = np.argmax(R)
    for i, R_val in enumerate(R):
        if not i == max_index:
            R_CI_lo[i] = R_val
            R_CI_hi[i] = R_val

    T_D, R_tot, R_tot_std, T_D_index, max_valid_index = plots.get_temporal_depth_and_R_tot(T, R)
    ax = axes[int(k/8)][k%8]
    # fig.set_size_inches(4, 3)
    ax.set_xscale('log')
    x_min = 0.005
    x_max = 5.
    ax.set_xlim((0.005, 5.))
    # ax.set_xticks(np.array([1, 10, 50]))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.spines['bottom'].set_bounds(0.005, 5.)
    ax.set_xticks([0.01, 0.1, 1.0])
    # if neuron_index == 104:
    #     ax.set_xlabel(r'past range $T$ [sec]')
    ##### y-axis ####
    # if neuron_index == 52:
    #     ax.set_ylabel(r'history dependence $R(T)$')
    max_val = np.amax(R)
    if max_val > 0.2:
        if max_val > 0.3:
            if max_val > 0.405:
                yrange = 0.41
                ymin = 0.1
                ax.set_ylim((0.1, .55))
                ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
                ax.spines['left'].set_bounds(.1, 0.5)
            else:
                yrange = 0.3
                ymin = 0.1
                ax.set_ylim((0.1, .45))
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
    # ax.set_xticks([])
    # ax.set_yticks([])
    if not int(k/8) == 17:
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        empty_string_labels = ['']*len(xlabels)
        ax.set_xticklabels(empty_string_labels)
    if not k%8 == 0:
        if not k==141:
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            empty_string_labels = ['']*len(ylabels)
            ax.set_yticklabels(empty_string_labels)

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)


    ax.plot(T, R, linewidth=1.2, color=green,
             label='Shuffling', zorder=3)
    ax.fill_between(T, R_CI_lo, R_CI_hi, facecolor=green, alpha=0.3)

    # shuffled
    ax.plot([T_D], [ymin], marker='d', markersize = 5., color=green,
             zorder=8)
    ax.plot([T_D], [R_tot], marker='x', markersize = 6., color=green,
             zorder=8)
    ax.axvline(x=T_D, ymax=(R_tot - ymin) / yrange, color=green,
                linewidth=0.5, linestyle='--')
    x = (np.log10(T_D) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot, xmax=x, color=green,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot], marker='d',markersize = 5., color=green,
             zorder=8)
    if k == 0:
        ax.text(0.007, R_tot + 0.06 *
                 R_tot, r'$\hat{R}_{\mathrm{tot}}$')
        ax.text(T_D + 0.2 * T_D, .005, r'$\hat{T}_D$')

    ax.plot(T[T_D_index:max_valid_index], np.zeros(max_valid_index-T_D_index)+R_tot, color = green, linewidth=1.5, linestyle='--')
    # R_tot_CI_lo = R_tot - 2 *  R_tot_std
    # R_tot_CI_hi = R_tot + 2 *  R_tot_std
    # ax.fill_between(T[T_D_index:max_valid_index], np.zeros(max_valid_index-T_D_index)+R_tot_CI_lo, np.zeros(max_valid_index-T_D_index)+R_tot_CI_hi, facecolor=green, alpha=0.3)

for j in np.arange(k+1,8*18):
    ax = axes[int(j/8)][j%8]
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    ax.spines['bottom'].set_bounds(0, 0)
    ax.spines['left'].set_bounds(0, 0)
    ax.set_xticks([])
    ax.set_yticks([])


fig.text(0.5, - 0.01, r'past range $T$ [sec]', ha='center', va='center', fontsize = 17)
fig.text(-0.01, 0.5, r'history dependence $R(T)$', ha='center', va='center', rotation='vertical',  fontsize = 17)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# fig.savefig("Subsampling_fixed_Cat.pdf")

plt.savefig('%s/Ropt_vs_T_all_V1.pdf'%(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')


plt.show()
plt.close()
