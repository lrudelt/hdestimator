import sys
import numpy as np

dataDir = '/data.nst/lucas/history_dependence/paper/retina_data/'

neuron_index = int(sys.argv[1])
validNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)
neuron = validNeurons[neuron_index]

spiketimes = np.loadtxt(
    '{}spks/spiketimes_neuron{}.dat'.format(dataDir, neuron))

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
