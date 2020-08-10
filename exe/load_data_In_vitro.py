import sys
import numpy as np

dataDir = '/data.nst/lucas/history_dependence/Paper/in_vitro_data/'
analysisDataDir = '/data.nst/lucas/history_dependence/Paper/in_vitro_data/analysis/'
# plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/In_vitro/'

neuron_index = int(sys.argv[1])

recordingNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)
neuron = recordingNeurons[neuron_index]

sample_rate = 24038.46169
spiketimes = np.loadtxt(
    '{}spks/spiketimes_neuron{}.dat'.format(dataDir, neuron))
spiketimes = spiketimes / sample_rate

rec_length = sys.argv[2]
if rec_length == '40min':
    T_rec = 2400.
if rec_length == '90min':
    T_rec = 5400.

T_0 = (spiketimes[2] + spiketimes[1]) / 2. - 10**(-4)
T_f = T_0 + T_rec

spiketimes = spiketimes - T_0
print(spiketimes[spiketimes < T_f])
