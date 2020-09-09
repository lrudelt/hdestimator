import os
import sys
from sys import exit, argv, path
from os.path import realpath, dirname
import pandas as pd
import yaml
import numpy as np

# ESTIMATOR_DIR = '{}/..'.format(dirname(realpath(__file__)))
ESTIMATOR_DIR = '/home/lucas/research/projects/history_dependence/hdestimator'
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))

if 'hde_glm' not in sys.modules:
    import hde_glm as glm

__version__ = "unknown"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

"""Run parameters"""
device = argv[1]
recorded_system = argv[2]
rec_length = argv[3]
# rec_length = '90min'
setup = argv[4]
# 'full_withCV', 'full',

if device == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

if device == 'cluster':
    sample_index = (int(os.environ['SGE_TASK_ID']) - 1)
else:
    sample_index = 0


def main():
    # Load settings
    with open('{}/settings/Simulation_glm.yaml'.format(ESTIMATOR_DIR), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)
    with open('{}/settings/Simulation_{}.yaml'.format(ESTIMATOR_DIR, setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)

    # Load the 900 minute simulated recording
    dataDir = '/home/lucas/research/projects/history_dependence/hdestimator/simulation_data/'
    spiketimes = np.loadtxt('{}spiketimes_constI_5ms.dat'.format(dataDir))

    # Preprocess spiketimes and compute binary counts for current spiking
    spiketimes, counts = glm.preprocess_spiketimes(spiketimes, glm_settings)

    # Load embedding parameters
    embedding_parameters_bbc, embedding_parameters_shuffling, analysis_num_str = glm.load_embedding_parameters(
        rec_length, sample_index, analysis_settings)
    temporal_depth_bbc, temporal_depth_shuffling, analysis_num_str = glm.get_temporal_depth(
        rec_length, sample_index, analysis_settings)
    # Compute glm for optimized embedding parameters for temporal depth, only if sample_index = 0 compute for all T
    if sample_index > 0:
        embedding_parameters_bbc = embedding_parameters_bbc[:,
                                                            embedding_parameters_bbc[0] == temporal_depth_bbc]
        embedding_parameters_shuffling = embedding_parameters_shuffling[:,
                                                                        embedding_parameters_shuffling[0] == temporal_depth_shuffling]
    # Compute history dependence with GLM for the same embeddings as found with bbc/shuffling
    glm_benchmark_bbc = glm.compute_benchmark_R(embedding_parameters_bbc,
                                                spiketimes, counts, glm_settings, regularization_method='bbc')
    glm_benchmark_shuffling = glm.compute_benchmark_R(embedding_parameters_shuffling,
                                                      spiketimes, counts, glm_settings, regularization_method='shuffling')
    # Save results to glm_benchmarks.csv
    glm.save_glm_benchmark_to_CSV(glm_benchmark_bbc, embedding_parameters_bbc,
                                  analysis_settings, analysis_num_str, regularization_method='bbc')
    glm.save_glm_benchmark_to_CSV(glm_benchmark_shuffling, embedding_parameters_shuffling,
                                  analysis_settings, analysis_num_str, regularization_method='shuffling')

    return EXIT_SUCCESS


if __name__ == "__main__":
    exit(main())
