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

T_0 = (spiketimes[2] + spiketimes[1]) / 2. - 10**(-4)
T_f = T_0 + T_rec

spiketimes = spiketimes - T_0
print(spiketimes[spiketimes < T_f])

# Number of neurons per cortical layer
# VISp2/3 14
# VISp4 33
# VISp5 56
# VISp6b 9
# VISp6a 30

# VISrl4 17
# VISrl5 32
# VISrl6a 14
# VISrl6b 2

# MOp2/3 32 31
# MOp5 31 17
# MOp6a 64 14
