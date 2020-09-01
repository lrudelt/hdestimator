from sys import argv
import os
from subprocess import call

"""Run parameters"""
device = argv[1]
recorded_system = argv[2]
rec_length = argv[3]
setup = argv[4]
# 'full_noCV', 'full', 'fast'

if device == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

if recorded_system == 'Simulation':
    if device == 'cluster':
        sample_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        sample_index = 0
else:
    if device == 'cluster':
        neuron_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        neuron_index = 0


"""Load spike data"""
codedirectory = '/home/lucas/research/projects/history_dependence/hdestimator'

"""Load data"""
if recorded_system == 'Simulation':
    load_script = '{}/exe/load_data_Simulation.py'.format(codedirectory)
    setting_file = '{}/settings/Simulation_{}.yaml'.format(
        codedirectory, setup)
else:
    load_script = '{}/exe/load_data_{}.py'.format(
        codedirectory, recorded_system)
    setting_file = '{}/settings/{}_{}.yaml'.format(
        codedirectory, recorded_system, setup)

program = '/home/lucas/anaconda2/envs/python3/bin/python'
# program='/home/lucas/anaconda3/bin/python -s'
script = '%s/estimate.py' % (codedirectory)


"""Compute estimates for different embeddings"""
if recorded_system == "Simulation":
    command = program + ' ' + load_script + ' ' + str(sample_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t hist -p -s ' + setting_file + \
        ' --label "{}-{}-{}"'.format(rec_length, setup, str(sample_index))
else:
    command = program + ' ' + load_script + ' ' + str(neuron_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t hist -p -s ' + setting_file + \
        ' -l "{}-{}-{}"'.format(rec_length, setup, str(neuron_index))

call(command, shell=True)
#
#
"""Create csv results files"""
if recorded_system == "Simulation":
    command = program + ' ' + load_script + ' ' + str(sample_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t csv -p -s ' + setting_file + \
        ' --label "{}-{}-{}"'.format(rec_length, setup, str(sample_index))
else:
    command = program + ' ' + load_script + ' ' + str(neuron_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t csv -p -s ' + setting_file + \
        ' -l "{}-{}-{}"'.format(rec_length, setup, str(neuron_index))

call(command, shell=True)
