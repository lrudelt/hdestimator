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
recorded_system = 'V1'
rec_length = '40min'
setup = 'fast_noCV'
# 'full'(only Simulation), 'full_noCV', 'fast' (only Experiments), 'one_bin' (only Experiments), 'fast_noCV' (only Experiments)
# recorded_system =  # 'V1' 142,  'Retina' 111, 'EC' 28, 'Culture' 48, 'Simulation' 10 (samples)

queue_option = 'qsub -q zal.q -t 1:142:1 -b y -j y -l h_vmem=6G -wd %s -o %s' % (
    codedirectory, outputdirectory)
#program = '%s/Mechanism/%s/x86_64/special -python' %(codedirectory, model)
program = '/home/lucas/anaconda2/envs/python3/bin/python'
script = '%s/emb_opt_analysis.py' % (codedirectory)
command = queue_option + ' ' + program + ' ' + \
    script + ' ' + device + ' ' + recorded_system + ' ' + rec_length + ' ' + setup

call(command, shell=True)
