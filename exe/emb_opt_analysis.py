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

if recording == 'Simulated':
    if sys.argv[1] == 'cluster':
        sample_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        sample_index = 0
    rec_length = sys.argv[3]
    setting = sys.argv[4]
    # 'full', 'fast'
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
    load_script = '%s/exe/load_data_Neuropixel.py'
    setting_file = '%s/Neuropixel_%s.yaml' % (codedirectory, setting)
if recording == 'Other':
    load_script = '%s/exe/load_data_{}.py'.format(area)
    setting_file = '%s/%s_%s.yaml' % (codedirectory, area, setting)
if recording == 'Simulated':
    load_script = '%s/exe/load_data_Simulated.py')
    setting_file='%s/Simulated_%s.yaml' % (codedirectory, setting)

program='/home/lucas/anaconda2/envs/python3/bin/python -s'
program='/home/lucas/anaconda3/bin/python -s'
script='%s/estimate.py' % (codedirectory)

if recording == "Simulated":
    command=program + ' ' + load_script + ' ' + sample_index + ' ' + rec_length + ' | ' +
               program + ' /dev/stdin -t hist -p -s ' + setting_file +
                   ' -l "{}-{}-{}"'.format(rec_length, setting, sample_index)
else:
    command=program + ' ' + load_script + ' ' + neuron_index + ' ' + rec_length + ' | ' +
               program + ' /dev/stdin -t hist -p -s ' + setting_file +
                   ' -l "{}-{}"'.format(rec_length, setting)

call(command, shell = True)


# TODO: Check if your function call is correct, and test on example data set with reduced setting
