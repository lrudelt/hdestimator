# Loading spiketimes for entorhinal cortex recording

dataDir = '/data.nst/lucas/history_dependence/Paper/Data_Simulated/'
analysisDataDir = '/data.nst/lucas/history_dependence/Paper/Data_Simulated/analysis/'
# plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/EC/'


"""Run parameters"""
sampling = sys.argv[1]  # 'iid', 'consecutive', sys.argv[2]
rec_length = sys.argv[2]
# Tp_index = (500 - 1) / 10
# sample_index = (500 - 1) % 10
Tp_index = (int(os.environ['SGE_TASK_ID']) - 1) / 10
sample_index = (int(os.environ['SGE_TASK_ID']) - 1) % 10


T_0 = 100. - 10**(-4) + T_rec * sample_index
T_f = T_0 + T_rec

spiketimes = np.loadtxt('{}spks/spiketimes_constI_5ms.dat'.format(dataDir))
spiketimes = spiketimes - T_0
