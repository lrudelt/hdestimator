# Loading spiketimes for neuropixel recordings
import sys
import numpy as np

dataDir = '/data.nst/lucas/history_dependence/Paper/Data_Neuropixel/Waksman/'
analysisDataDir = '/data.nst/lucas/history_dependence/Data_Neuropixel/Waksman/analysis/'
# plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/Neuropixel/'

validNeuronsAreas = np.load('{}validNeuronsAreas.npy'.format(
    dataDir), allow_pickle=True).item()
analyzedNeurons = np.load('{}analyzedNeurons.npy'.format(dataDir))
areaLayers = {'primaryVisualCortex': ['VISp23', 'VISp4', 'VISp5', 'VISp6b', 'VISp6a'], 'rostrolateralArea': [
    'VISrl4', 'VISrl5', 'VISrl6b', 'VISrl6a'], 'primaryMotorCortex': ['MOp5', 'MOp6a', 'MOp23']}

recordingNeurons = []
for layer in areaLayers[area]:
    for neuron in validNeuronsAreas[layer]:
        recordingNeurons += [neuron]

neuron_index = int(sys.argv[1])
neuron = recordingNeurons[neuron_index]
spiketimes = np.load('{}spks/spikes-{}-{}.npy'.format(dataDir,
                                                      neuron[0], neuron[1]))

rec_length = sys.argv[2]
if rec_length == '40min':
    T_rec = 2400.
if rec_length == '90min':
    T_rec = 5400.

# Add 5 seconds to make sure that only spikes with sufficient spiking history are considered
T_0 = spiketimes[0] + 5.
T_f = T_0 + T_rec

spiketimes = spiketimes - T_0
print(spiketimes[spiketimes < T_f])

# Number of neurons per cortical layer
# VISp2/3 14
# VISp4 33
# VISp5 56
# VISp6b 9
# VISp6a 30
