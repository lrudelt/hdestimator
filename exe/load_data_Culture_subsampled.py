import sys
import numpy as np

dataDir = '/data.nst/lucas/history_dependence/paper/culture_data/'

run_index = int(sys.argv[1])
rec_length = sys.argv[2]

rec_lengths = {'1min': 60., '3min': 180., '5min': 300.,
               '10min': 600., '20min': 1200., '45min': 2700., '90min': 5400.}

rec_lengths_Nsamples = {'1min': 10, '3min': 10, '5min': 10,
               '10min': 8, '20min': 4, '45min': 2}

N_neurons = 10
N_samples = rec_lengths_Nsamples[rec_length]
T_rec = rec_lengths[rec_length]

neuron_index = int(run_index/N_samples)
sample_index = run_index%N_samples

validNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)

np.random.seed(41)
neuron_selection = np.random.choice(len(validNeurons), N_neurons,  replace = False)
neuron = validNeurons[neuron_selection][neuron_index]

sample_rate = 24038.46169
spiketimes = np.loadtxt(
    '{}spks/spiketimes_neuron{}.dat'.format(dataDir, neuron))
spiketimes = spiketimes / sample_rate

# Add 5 seconds to make sure that only spikes with sufficient spiking history are considered
T_0 = spiketimes[0] + 5.
T_step = 4800. / N_samples

T_0 = T_0 + sample_index * T_step

spiketimes = spiketimes - T_0
spiketimes = spiketimes[spiketimes > 0]
spiketimes = spiketimes[spiketimes < T_rec]
print(*spiketimes, sep='\n')
