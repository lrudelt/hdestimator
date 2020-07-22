
dataDir = '/data.nst/lucas/history_dependence/Data_Retina/'
analysisDataDir = '/data.nst/lucas/history_dependence/Data_Retina/dataAnalysis/'
plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/Retina/'

recordingNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)
neuron = recordingNeurons[neuron_index]

spiketimes = np.loadtxt(
    '{}spks/spiketimes_neuron{}.dat'.format(dataDir, neuron))
# data['data'][0][0][3][0][0][0][0][0] / sampling_rate
T_0 = (spiketimes[2] + spiketimes[1]) / 2. - 10**(-4)
# T_f = 4925.5381  # data['data'][0][0][3][0][0][1][1][0] / sampling_rate

# Description
# print(data['data'][0][0][0])

# Date
# print(data['data'][0][0][1])

# Sampling rate
# print(data['data'][0][0][2][0][0][0])

# Neuron list
# neurons = data['data'][0][0][2][0][0][2][0]
# N_neurons = neurons[-1]

# for neuron in range(N_neurons):
# spiketimes = data['data'][0][0][2][0][0][1][0][neuron][0] / sampling_rate
# np.savetxt('../../Data/Retina/experimental_Data/spiketimes_neuron%d.dat' %
# neuron, spiketimes)

# spiketimes
# spiketimes = data['data'][0][0][2][0][0][1][0][neuron][0] / sampling_rate


# Compute valid neurons with rate > 0.5 Hz
# valid_neurons=[]
# for neuron in range(N_neurons):
#     spiketimes=data['data'][0][0][2][0][0][1][0][neuron][0]
#     rate=spiketimes.size/T
#     if rate>=0.5:
#         print rate
#         valid_neurons+=[neuron]

# Full data
# print(data['data'][0][0][2][0][0][1])

# Short data, but not really sure what this is. Spiketimes are not the same
# print(data['data'][0][0][2][0][0][3])
