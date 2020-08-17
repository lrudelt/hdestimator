import sys
import numpy as np
from scipy.io import loadmat
# Loading spiketimes for entorhinal cortex recording

dataDir = '/data.nst/lucas/history_dependence/Paper/EC_data/'
analysisDataDir = '/data.nst/lucas/history_dependence/Paper/EC_data/analysis/'
# plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/EC/'

neuron_index = int(sys.argv[1])
recordingNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)
neuron = recordingNeurons[neuron_index]

sptimes_raw = loadmat('{}spks/ec014.277.spike_ch.mat'.format(dataDir))
sample_rate = 20000.
sptimes = sptimes_raw['sptimes'][0] / sample_rate
spiketimes = sptimes[neuron].flatten()

rec_length = sys.argv[2]
if rec_length == '40min':
    T_rec = 2400.
if rec_length == '90min':
    T_rec = 5400.

T_0 = (spiketimes[2] + spiketimes[1]) / 2. - 10**(-4)

spiketimes = spiketimes - T_0
spiketimes = spiketimes[spiketimes > 0]
spiketimes = spiketimes[spiketimes < T_rec]
print(*spiketimes, sep='\n')
