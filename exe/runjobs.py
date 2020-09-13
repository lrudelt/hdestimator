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
recorded_system = 'Culture'
rec_length = '90min'
setup = 'full_shuffling'
# 'full', 'full_withCV' (only Simulation), 'full_bbc' (test of the old BBC criterion), 'onebin' (only Experiments), 'fivebins' (max five bins, only Experiments)
# recorded_system =  # 'V1' 142,  'Retina' 111, 'EC' 28, 'Culture' 48, 'Simulation' 10 (samples)


queue_option = 'qsub -q rostam.q -t 1:48:1 -b y -j y -l h_vmem=6G -wd %s -o %s' % (
    codedirectory, outputdirectory)
#program = '%s/Mechanism/%s/x86_64/special -python' %(codedirectory, model)
program = '/home/lucas/anaconda2/envs/python3/bin/python'
script = '%s/confidence_intervals.py' % (codedirectory)
command = queue_option + ' ' + program + ' ' + \
    script + ' ' + device + ' ' + recorded_system + ' ' + rec_length + ' ' + setup

call(command, shell=True)


# for rec_length in ['1min', '3min', '5min', '10min', '20min']:
#     queue_option = 'qsub -q rostam.q -t 1:30:1 -b y -j y -l h_vmem=6G -wd %s -o %s' % (
#         codedirectory, outputdirectory)
#     #program = '%s/Mechanism/%s/x86_64/special -python' %(codedirectory, model)
#     program = '/home/lucas/anaconda2/envs/python3/bin/python'
#     script = '%s/emb_opt_analysis.py' % (codedirectory)
#     command = queue_option + ' ' + program + ' ' + \
#         script + ' ' + device + ' ' + recorded_system + ' ' + rec_length + ' ' + setup
#
#     call(command, shell=True)
#
# for rec_length in ['45min', '90min']:
#     queue_option = 'qsub -q rostam.q -t 1:10:1 -b y -j y -l h_vmem=6G -wd %s -o %s' % (
#         codedirectory, outputdirectory)
#     #program = '%s/Mechanism/%s/x86_64/special -python' %(codedirectory, model)
#     program = '/home/lucas/anaconda2/envs/python3/bin/python'
#     script = '%s/emb_opt_analysis.py' % (codedirectory)
#     command = queue_option + ' ' + program + ' ' + \
#         script + ' ' + device + ' ' + recorded_system + ' ' + rec_length + ' ' + setup
#
#     call(command, shell=True)
