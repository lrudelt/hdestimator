import sys
from subprocess import call
if sys.argv[1] == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

"""Run parameters"""
recording = 'Simulation'
# recording = sys.argv[2]
# 'Neuropixel', 'Other', 'Simulation'

rec_length = '1min'
setting = 'full_noCV'
sample_index = 0

if recording == 'Simulation':
    if sys.argv[1] == 'cluster':
        sample_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        sample_index = 0
    rec_length = sys.argv[3]
    setting = sys.argv[4]
    # 'full_noCV', 'full'
else:
    if sys.argv[1] == 'cluster':
        neuron_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        neuron_index = 0
    area = sys.argv[3]
    # primaryVisualCortex, rostrolateralArea, EC, Retina, In_vitro
    rec_length = sys.argv[4]
    # rec_length = '40min', '90min', 'full'
    setting = sys.argv[5]
    # 'full_noCV', 'full', 'fast'

"""Load spike data"""
codedirectory = '/home/lucas/research/projects/history_dependence/hdestimator'
"""Load data"""
if recording == 'Neuropixel':
    load_script = '{}/exe/load_data_Neuropixel.py'.format(codedirectory)
    setting_file = '{}/settings/Neuropixel_{}.yaml'.format(
        codedirectory, setting)
if recording == 'Other':
    load_script = '{}/exe/load_data_{}.py'.format(codedirectory, area)
    setting_file = '{}/settings/{}_{}.yaml'.format(
        codedirectory, area, setting)
if recording == 'Simulation':
    load_script = '{}/exe/load_data_Simulation.py'.format(codedirectory)
    setting_file = '{}/settings/Simulation_{}.yaml'.format(
        codedirectory, setting)

program = '/home/lucas/anaconda2/envs/python3/bin/python'
# program='/home/lucas/anaconda3/bin/python -s'
script = '%s/estimate.py' % (codedirectory)

if recording == "Simulation":
    command = program + ' ' + load_script + ' ' + str(sample_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t hist -p -s ' + setting_file + \
        ' --label "{}-{}-{}"'.format(rec_length, setting, str(sample_index))
else:
    command = program + ' ' + load_script + ' ' + str(neuron_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t hist -p -s ' + setting_file + \
        ' -l "{}-{}"'.format(rec_length, setting)

call(command, shell=True)

# TODO: Check if your function call is correct, and test on example data set with reduced setting

# command = program + ' ' + load_script + ' ' + str(sample_index) + ' ' + rec_length + ' | ' + program + ' ' + script + ' /dev/stdin -t plots -p -s ' + setting_file + \
#     ' -l "{}-{}-{}"'.format(rec_length, setting, str(sample_index))
