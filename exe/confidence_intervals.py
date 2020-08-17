from sys import argv
import os
from subprocess import call

"""Run parameters"""
device = argv[1]
recorded_system = argv[2]
rec_length = argv[3]
setting = argv[4]
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
        codedirectory, setting)
else:
    load_script = '{}/exe/load_data_{}.py'.format(
        codedirectory, recorded_system)
    setting_file = '{}/settings/{}_{}.yaml'.format(
        codedirectory, recorded_system, setting)

program = '/home/lucas/anaconda2/envs/python3/bin/python'
# program='/home/lucas/anaconda3/bin/python -s'
script = '%s/estimate.py' % (codedirectory)

# For the first sample, produce confidence intervals and plots
if recorded_system == "Simulation":
    if sample_index == 0:
        command = program + ' ' + load_script + ' ' + str(sample_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t conf -p -s ' + setting_file + \
            ' --label "{}-{}-{}"'.format(rec_length,
                                         setting, str(sample_index))
        call(command, shell=True)
        command = program + ' ' + load_script + ' ' + str(sample_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t csv -p -s ' + setting_file + \
            ' --label "{}-{}-{}"'.format(rec_length,
                                         setting, str(sample_index))
        call(command, shell=True)
        command = program + ' ' + load_script + ' ' + str(sample_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t plots -p -s ' + setting_file + \
            ' --label "{}-{}-{}"'.format(rec_length,
                                         setting, str(sample_index))
        call(command, shell=True)
