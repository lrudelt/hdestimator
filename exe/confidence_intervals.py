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
    run_index = (int(os.environ['SGE_TASK_ID']) - 1)
else:
    run_index = 96

codedirectory = '/home/lucas/research/projects/history_dependence/hdestimator'


load_script = '{}/exe/load_data_{}.py'.format(
    codedirectory, recorded_system)
# setting_file = '{}/settings/{}_{}_withCI.yaml'.format(
#      codedirectory, recorded_system, setup)
setting_file = '{}/settings/{}_{}.yaml'.format(
     codedirectory, recorded_system, setup)

program = '/home/lucas/anaconda2/envs/python3/bin/python'
# program='/home/lucas/anaconda3/bin/python -s'
script = '%s/estimate.py' % (codedirectory)

# For the first sample, produce confidence intervals and plots
command = program + ' ' + load_script + ' ' + str(run_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t conf -p -s ' + setting_file + \
    ' --label "{}-{}-{}"'.format(rec_length,
                                 setup, str(run_index))
call(command, shell=True)

command = program + ' ' + load_script + ' ' + str(run_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t csv -p -s ' + setting_file + \
    ' --label "{}-{}-{}"'.format(rec_length,
                                 setup, str(run_index))
call(command, shell=True)
command = program + ' ' + load_script + ' ' + str(run_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t plots -p -s ' + setting_file + \
    ' --label "{}-{}-{}"'.format(rec_length,
                                 setup, str(run_index))
call(command, shell=True)
