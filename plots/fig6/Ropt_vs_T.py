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
fig = 'fig5'
PLOTTING_DIR = '/home/lucas/research/papers/history_dependence/arXiv/figs/{}'.format(
    fig)

"""Load data"""
recorded_system = 'V1'
setup = 'fivebins'
rec_length = '40min'
neuron_index = 62
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
neuron = validNeurons[neuron_index]


R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
    recorded_system, rec_length, neuron_index, setup, ESTIMATOR_DIR, regularization_method = 'shuffling')
R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
# T_D_new, R_tot_new, T_D_index, max_valid_index = plots.get_temporal_depth_and_R_tot(T, R)

"""Plot"""

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '16.0'
matplotlib.rcParams['xtick.labelsize'] = '16'
matplotlib.rcParams['ytick.labelsize'] = '16'
matplotlib.rcParams['legend.fontsize'] = '16'
matplotlib.rcParams['axes.linewidth'] = 0.6
# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

##### x-axis ####

fig, ((ax1)) = plt.subplots(1, 1, figsize=(3.5, 2.8))

# fig.set_size_inches(4, 3)
ax1.set_xscale('log')
x_min = 0.005
x_max = 5.
ax1.set_xlim((0.005, 5.))
# ax1.set_xticks(np.array([1, 10, 50]))
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.spines['bottom'].set_bounds(0.005, 5.)
ax1.set_xticks([0.01, 0.1, 1.0])
if neuron_index == 104:
    ax1.set_xlabel(r'past range $T$ [sec]')
else:
    ax1.set_xlabel(r'past range $T$ [sec]',alpha=0.0)

##### y-axis ####
if neuron_index == 52:
    ax1.set_ylabel(r'history dependence $R(T)$')
else:
    ax1.set_ylabel(r'history dependence $R(T)$',alpha=0.0)
max_val = np.amax(R)
if max_val > 0.2:
    if max_val > 0.3:
        if max_val > 0.405:
            yrange = 0.41
            ymin = 0.1
            ax1.set_ylim((0.1, .51))
            ax1.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
            ax1.spines['left'].set_bounds(.1, 0.5)
        else:
            yrange = 0.3
            ymin = 0.1
            ax1.set_ylim((0.1, .4))
            ax1.set_yticks([0.1, 0.2, 0.3, 0.4])
            ax1.spines['left'].set_bounds(.1, 0.4)
    else:
        yrange = 0.3
        ymin = 0.0
        ax1.set_ylim((0.0, .3))
        ax1.set_yticks([0.0, 0.1, 0.2, 0.3])
        ax1.spines['left'].set_bounds(.0, 0.3)
else:
    yrange = 0.2
    ymin = 0.0
    ax1.set_ylim((0.0, .2))
    ax1.set_yticks([0.0, 0.1, 0.2])
    ax1.spines['left'].set_bounds(.0, 0.2)

##### Unset Borders #####
ax1.spines['top'].set_bounds(0, 0)
ax1.spines['right'].set_bounds(0, 0)

if not neuron_index == 52:
    labels = [item.get_text() for item in ax.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    ax1.set_yticklabels(empty_string_labels)

ax1.plot(T, R, linewidth=1.2, color=green,
         label='Shuffling', zorder=3)
ax1.fill_between(T, R_CI_lo, R_CI_hi, facecolor=green, alpha=0.3)


# shuffled
ax1.plot([T_D], [ymin], marker='d', markersize = 5., color=green,
         zorder=8)
ax1.plot([T_D], [R_tot], marker='x', markersize = 6., color=green,
         zorder=8)
ax1.axvline(x=T_D, ymax=(R_tot - ymin) / yrange, color=green,
            linewidth=0.5, linestyle='--')
x = (np.log10(T_D) - np.log10(x_min)) / \
    (np.log10(x_max) - np.log10(x_min))
ax1.axhline(y=R_tot, xmax=x, color=green,
            linewidth=0.5, linestyle='--')
ax1.plot([x_min], [R_tot], marker='d',markersize = 5., color=green,
         zorder=8)
ax1.text(0.007, R_tot + 0.06 *
         R_tot, r'$\hat{R}_{\mathrm{tot}}$')
ax1.text(T_D + 0.2 * T_D, .005, r'$\hat{T}_D$')

ax1.plot(T[T_D_index:max_valid_index], np.zeros(max_valid_index-T_D_index)+R_tot, color = green,linestyle='--')

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
# fig.savefig("Subsampling_fixed_Cat.pdf")

plt.savefig('%s/Ropt_vs_T_neuron%d-%d.pdf'%(PLOTTING_DIR,neuron[0],neuron[1]),
            format="pdf", bbox_inches='tight')


plt.show()
plt.close()
