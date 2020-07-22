import sys
import numpy as np

dataDir = '/data.nst/lucas/history_dependence/Paper/Data_Simulated/'
analysisDataDir = '/data.nst/lucas/history_dependence/Paper/Data_Simulated/analysis/'
# plottingDataDir = '/home/lucas/research/projects/history_dependence/data_plotting/EC/'

sample_index = int(sys.argv[1])
rec_length = sys.argv[2]

rec_lengths = {'1min': 60., '3min': 180., '5min': 300., '10min': 600.,
                          '20min', 1200., '45min':2700., '90min': 5400.}

T_rec = rec_lengths[rec_length]

T_0 = 100. - 10**(-4) + T_rec * sample_index
T_f = T_0 + T_rec

spiketimes = np.loadtxt('{}spks/spiketimes_constI_5ms.dat'.format(dataDir))
spiketimes = spiketimes - T_0
print(spiketimes[spiketimes < T_f])
