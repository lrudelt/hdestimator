import sys
import numpy as np
from scipy.io import loadmat
# Loading spiketimes for entorhinal cortex recording

dataDir = '/data.nst/lucas/history_dependence/paper/EC_data/'
# plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/EC/'

neuron_index = int(sys.argv[1])
validNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)
neuron = validNeurons[neuron_index]

sptimes_raw = loadmat('{}spks/ec014.277.spike_ch.mat'.format(dataDir))
sample_rate = 20000.
sptimes = sptimes_raw['sptimes'][0] / sample_rate
spiketimes = sptimes[neuron].flatten()

rec_length = sys.argv[2]
if rec_length == '40min':
    T_rec = 2400.
if rec_length == '90min':
    T_rec = 5400.

# Add 5 seconds to make sure that only spikes with sufficient spiking history are considered
T_0 = spiketimes[0] + 5.

spiketimes = spiketimes - T_0
spiketimes = spiketimes[spiketimes > 0]
spiketimes = spiketimes[spiketimes < T_rec]
print(*spiketimes, sep='\n')
