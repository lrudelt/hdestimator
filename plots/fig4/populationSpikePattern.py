"""Functions"""
from scipy.io import loadmat
import seaborn.apionly as sns
from scipy.optimize import bisect
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import matplotlib
import numpy as np
import sys
sys.path.append('../../Scripts/Functions')

##### Plot params #####

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '13.0'
matplotlib.rcParams['xtick.labelsize'] = '13.0'
matplotlib.rcParams['ytick.labelsize'] = '13.0'
matplotlib.rcParams['legend.fontsize'] = '13.0'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ax = plt.subplots(1, 1, figsize=(3., 2.8))

##### Unset Borders #####

for side in ['right', 'top', 'left']:
    ax.spines[side].set_visible(False)

##### remove axis ticks #####

ax.xaxis.set_ticks_position('none')  # tick markers
ax.yaxis.set_ticks_position('none')

##### unset ticks and labels #####
plt.xticks([])  # labels
plt.yticks([])
plt.setp([a.get_xticklabels() for a in [ax]], visible=False)
plt.setp([a.get_yticklabels() for a in [ax]], visible=False)

"""Load spike data"""
recorded_system = sys.argv[1]

"""For Retina data"""
if recorded_system == 'Retina':
    data_path = "/data.nst/lucas/history_dependence/paper/retina_data/spks"

    ax.set_title("salamander retina", fontsize=17)

    ##### x-axis ####
    ax.set_xlim((52, 62))
    ax.spines['bottom'].set_bounds(60, 62)
    ax.text(60.7, -14, '2s')

    ##### y-axis ####
    ax.set_ylabel(r'neuron \#')
    ax.set_ylim((-1.5, 152.5))
    ax.spines['left'].set_bounds(1, 152)

    ##### plot spikes ####
    for neuron in range(152):
        spiketimes = np.loadtxt(
            '%s/spiketimes_neuron%d.dat' % (data_path, neuron))
        ax.scatter(spiketimes[spiketimes < 63.], np.zeros(len(spiketimes[spiketimes < 63.])) +
                   neuron + 1, s=0.2, color='0', zorder=1)

"""For in vivo data"""
if recorded_system == 'EC':
    data_path = "/data.nst/lucas/history_dependence/paper/EC_data/spks"
    data = loadmat('%s/ec014.277.spike_ch.mat' % data_path)
    sample_rate = 20000.  # in seconds
    sptimes = data['sptimes'][0] / sample_rate
    singleunit = data['singleunit'][0]
    sptimes_multi = []
    sptimes_single = []
    neurons_multi = []
    neurons_single = []
    for i in range(85):
        if singleunit[i] == 0:
            sptimes_multi = np.append(sptimes_multi, sptimes[i])
            neurons_multi = np.append(
                neurons_multi, np.zeros(sptimes[i].size) + i)
        else:
            if len(sptimes[i]) / (sptimes[i][-1] - sptimes[i][0]) < 10:
                sptimes_single = np.append(sptimes_single, sptimes[i])
                neurons_single = np.append(
                    neurons_single, np.zeros(sptimes[i].size) + i)
    neurons_single = neurons_single + 1
    neurons_multi = neurons_multi + 1

    ax.set_title("rat entorhinal cortex", fontsize=17)

    ##### x-axis ####
    ax.set_xlim((52, 62))
    ax.spines['bottom'].set_bounds(60, 62)
    ax.text(60.7, -8, '2s')

    ##### y-axis ####
    ax.set_ylabel(r'neuron \#')
    ax.set_ylim((0.5, 85.5))
    ax.spines['left'].set_bounds(1, 85)

    ##### plot spikes of single units only ####
    ax.scatter(sptimes_single[sptimes_single < 63.],
               neurons_single[sptimes_single < 63.], s=0.2, color='0', zorder=1)


"""For in vitro data"""
if recorded_system == 'Culture':
    data_path = "/data.nst/lucas/history_dependence/paper/culture_data/spks"

    ax.set_title("rat entorhinal cortex", fontsize=17)

    ##### x-axis ####
    ax.set_xlim((52, 62))
    ax.spines['bottom'].set_bounds(60, 62)
    ax.text(60.7, -7, '2s')

    ##### y-axis ####
    ax.set_ylabel(r'neuron \#')
    ax.set_ylim((-1.5, 60.5))
    ax.spines['left'].set_bounds(1, 60)

    ##### plot spikes ####
    sample_rate = 24038.46169
    T_0 = 20 - 0.000001
    for neuron in np.arange(1, 61):
        spiketimes = np.loadtxt('%s/spiketimes_neuron%d.dat' %
                                (data_path, neuron)) / sample_rate - T_0
        ax.scatter(spiketimes[spiketimes < 63.], np.zeros(
            len(spiketimes[spiketimes < 63.])) + neuron, s=0.2, color='0', zorder=1)

if recorded_system == 'V1':
    area = 'primaryVisualCortex'
    data_path = "/data.nst/lucas/history_dependence/paper/neuropixel_data/Waksman"
    validNeuronsAreas = np.load('{}/validNeuronsAreas.npy'.format(
        data_path), allow_pickle=True).item()
    areaLayers = {'primaryVisualCortex': ['VISp23', 'VISp4', 'VISp5', 'VISp6b', 'VISp6a'], 'rostrolateralArea': [
        'VISrl4', 'VISrl5', 'VISrl6b', 'VISrl6a'], 'primaryMotorCortex': ['MOp5', 'MOp6a', 'MOp23']}

    ax.set_title("mouse primary visual cortex", fontsize=17)

    ##### x-axis ####
    ax.set_xlim((52, 62))
    ax.spines['bottom'].set_bounds(60, 62)
    ax.text(60.7, -13, '2s')

    ##### y-axis ####
    ax.set_ylabel(r'neuron \#')
    ax.set_ylim((-1.5, 142.5))
    ax.spines['left'].set_bounds(1, 142)

    ##### plot spikes of analyzed neurons only ####
    neuron_ID = 1
    for layer in areaLayers[area]:
        for neuron in validNeuronsAreas[layer]:
            spiketimes = np.load('{}/spks/spikes-{}-{}.npy'.format(data_path,
                                                                   neuron[0], neuron[1]))
            T_0 = spiketimes[0]
            spiketimes = spiketimes - T_0
            ax.scatter(spiketimes[spiketimes < 63.], np.zeros(len(spiketimes[spiketimes < 63.])) +
                       neuron_ID, s=0.2, color='0', zorder=1)
            neuron_ID += 1

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('/data.nst/lucas/history_dependence/paper/plots/fig4/%sSpikePattern.pdf' % recorded_system,
            format="pdf", bbox_inches='tight')
plt.savefig('/data.nst/lucas/history_dependence/paper/plots/fig4/%sSpikePattern.png' % recorded_system,
            format="png", dpi=400, bbox_inches='tight')

plt.show()
plt.close()
