
dataDir = '/data.nst/lucas/history_dependence/Data_In_vitro/'
analysisDataDir = '/data.nst/lucas/history_dependence/Data_In_vitro/dataAnalysis/'
plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/In_vitro/'

recordingNeurons = np.load(
    '{}validNeurons.npy'.format(dataDir)).astype(int)
neuron = recordingNeurons[neuron_index]

sample_rate = 24038.46169
spiketimes = np.loadtxt(
    '{}spks/spiketimes_neuron{}.dat'.format(dataDir, neuron))
spiketimes = spiketimes / sample_rate
T_0 = (spiketimes[2] + spiketimes[1]) / 2. - 10**(-4)
