#!/usr/bin/python
from subprocess import call
import sys
import time
import os
import collections

hostname = 'lucas'
codedirectory = '/home/lucas/research/projects/history_dependence/hdestimator/exe'
outputdirectory = '/data.nst/lucas/history_dependence/cluster_output/'

device = 'cluster'
recorded_system = 'EC'
setup = 'full_shuffling_subsampled'
N_neurons = 10
rec_lengths_Nsamples = {'1min': 10, '3min': 10, '5min': 10,
               '10min': 8, '20min': 4, '45min': 2}
# 'full', 'full_withCV' (only Simulation), 'full_bbc' (test of the old BBC criterion), 'onebin' (only Experiments), 'fivebins' (max five bins, only Experiments)
# recorded_system =  # 'V1' 142,  'Retina' 111, 'EC' 28, 'Culture' 48, 'Simulation' 10 (samples)

for rec_length in ['3min','5min','10min','20min','45min']:
	N_samples = rec_lengths_Nsamples[rec_length]
	N_runs = N_neurons * N_samples
	queue_option = 'qsub -q rostam.q -t 1:{}:1 -b y -j y -l h_vmem=6G -wd {} -o {}'.format(
         N_runs, codedirectory, outputdirectory)
        #program = '%s/Mechanism/%s/x86_64/special -python' %(codedirectory, model)
	program = '/home/lucas/anaconda2/envs/python3/bin/python'
	script = '%s/emb_opt_analysis_subsampled.py' % (codedirectory)
	command = queue_option + ' ' + program + ' ' + \
 	script + ' ' + device + ' ' + recorded_system + ' ' + rec_length + ' ' + setup
	call(command, shell=True)
