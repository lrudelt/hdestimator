import sys
import numpy as np

dataDir = '/data.nst/lucas/history_dependence/Paper/retina_data/'
analysisDataDir = '/data.nst/lucas/history_dependence/Paper/retina_data/analysis/'
# plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/EC/'

neuron_index = int(sys.argv[1])
recordingNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)
neuron = recordingNeurons[neuron_index]

spiketimes = np.loadtxt(
    '{}spks/spiketimes_neuron{}.dat'.format(dataDir, neuron))

rec_length = sys.argv[2]
if rec_length == '40min':
    T_rec = 2400.
if rec_length == '90min':
    T_rec = 5400.

T_0 = (spiketimes[2] + spiketimes[1]) / 2. - 10**(-4)
T_f = T_0 + T_rec

spiketimes = spiketimes - T_0
print(spiketimes[spiketimes < T_f])
