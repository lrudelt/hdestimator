import sys
from subprocess import call
if sys.argv[1] == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

"""Run parameters"""
recording = sys.argv[2]
# 'Neuropixel', 'Other', 'Simulated'
area = sys.argv[3]
# recording = 'primaryMotorCortex'  # primaryVisualCortex, rostrolateralArea, EC, Retina, In_vitro
rec_length = sys.argv[4]
# rec_length = '40min', '90min', 'full'
setting = sys.argv[5]
# 'full', 'fast'
if recording == 'Simulated':
    sample_index = (int(os.environ['SGE_TASK_ID']) - 1)
else:
    neuron_index = (int(os.environ['SGE_TASK_ID']) - 1)

"""Load spike data"""
codedirectory = '/home/lucas/research/projects/history_dependence/hdestimator'
"""Load data"""
if recording == 'Neuropixel':
    load_script = '%s/exe/load_data_Neuropixel.py'
    setting_file = '%s/%s_Neuropixel_settings.yaml' % (codedirectory, setting)
if recording == 'Other':
    load_script = '%s/exe/load_data_{}.py'.format(area)
    setting_file = '%s/%s_%s_settings.yaml' % (codedirectory, setting, area)
if recording == 'Simulated':
    load_script = '%s/exe/load_data_Simulated.py')
    setting_file = '%s/%s_Simulated_settings.yaml' % (codedirectory, setting)

program ='/home/lucas/anaconda2/envs/python3/bin/python -s'
program = '/home/lucas/anaconda3/bin/python -s'
script = '%s/estimate.py' % (codedirectory)

if recording == "Simulated":
    command =  program + ' ' + load_script + ' ' + rec_length + ' | ' + \
               program + ' /dev/stdin -t hist -p -s ' + setting_file + ' -l "{}-{}-{}"'.format(rec_length, setting, sample_index)
else:
    command =  program + ' ' + load_script + ' ' + rec_length + ' | ' + \
               program + ' /dev/stdin -t hist -p -s ' + setting_file + ' -l "{}-{}"'.format(rec_length, setting)
# Make sure that the threads are on a single core

call(command, shell=True)


# TODO: Write out load_data_scripts (idea for Simulated: choose 10 evenly spaced consecutive samples from the whole recordin, but how to choose cross-validated? Only perform embedding optimization on the first half and evaluate on the full recording, or take the next recording in line)
# TODO: Write settings files
# TODO: Check if your function call is correct, and test on example data set with reduced setting
