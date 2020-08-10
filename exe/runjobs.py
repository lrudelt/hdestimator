#!/usr/bin/python
from subprocess import call
import sys
import time
import os
import collections
import numpy as np
hostname = 'lucas'
codedirectory = '/home/lucas/research/projects/history_dependence/hdestimator/exe'
outputdirectory = '/data.nst/lucas/history_dependence/cluster_output/'

device = 'cluster'
recording = 'Simulation'
# 'Other', 'Neuropixel', 'Simulation'
area = 'Retina'
rec_length = '1min'
setting = 'full_noCV'
# 'fast'
# area = 'primaryMotorCortex' 127  # primaryVisualCortex 142, rostrolateralArea 65, Retina 111, EC 28, In_vitro 48
# dataDir = '/data.nst/lucas/history_dependence/Data_Neuropixel/Waksman/{}/'.format(
#     area)
# valid_neurons = np.load('{}validNeurons.npy'.format(dataDir))
# Nneurons = len(valid_neurons)

queue_option = 'qsub -q rostam.q  -t 1:111:1 -b y -j y -l h_vmem=6G -wd %s -o %s' % (
    codedirectory, outputdirectory)
#program = '%s/Mechanism/%s/x86_64/special -python' %(codedirectory, model)
program = '/home/lucas/anaconda2/bin/python -s'
script = '%s/emb_opt_analysis.py' % (codedirectory)
if recording == 'Simulated':
    command = queue_option + ' ' + program + ' ' + \
        script + ' ' + device + ' ' + recording + \
        ' ' + rec_length + ' ' + setting
else:
    command = queue_option + ' ' + program + ' ' + \
        script + ' ' + device + ' ' + recording + ' ' + \
        area + ' ' + ' ' + rec_length + ' ' + setting
# Make sure that the threads are on a single core

call(command, shell=True)
