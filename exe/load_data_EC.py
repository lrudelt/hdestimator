# Loading spiketimes for entorhinal cortex recording

dataDir = '/data.nst/lucas/history_dependence/Data_EC/'
analysisDataDir = '/data.nst/lucas/history_dependence/Data_EC/dataAnalysis/'
plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/EC/'

recordingNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)
neuron = recordingNeurons[neuron_index]

sptimes_raw = loadmat('{}spks/ec014.277.spike_ch.mat'.format(dataDir))
sample_rate = 20000.
sptimes = sptimes_raw['sptimes'][0] / sample_rate
spiketimes = sptimes[neuron].flatten()
T_0 = (spiketimes[2] + spiketimes[1]) / 2. - 10**(-4)
