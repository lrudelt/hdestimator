import os
import sys
from sys import exit, argv, path
from os.path import realpath, dirname
import csv
import yaml
import numpy as np

# ESTIMATOR_DIR = '{}/..'.format(dirname(realpath(__file__)))
ESTIMATOR_DIR = '/home/lucas/research/projects/history_dependence/hdestimator/'
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
setup = argv[4]
# device = 'Desktop'
# recorded_system = 'Simulation'
# rec_length = '90min'
# setup = 'full'


if device == 'cluster':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'


def main_Simulation():
    # Get run index for computation on the cluster
    if device == 'cluster':
        past_range_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        past_range_index = 1
    # Load settings
    with open('{}/settings/Simulation_glm.yaml'.format(ESTIMATOR_DIR), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)
    if past_range_index == 0:
        analysis_dir = glm_settings['ANALYSIS_DIR']
        glm_csv_file_name = '{}/glm_estimates_cross_validated_new.csv'.format(
            analysis_dir)
        with open(glm_csv_file_name, 'w', newline='') as glm_csv_file:
            writer = csv.DictWriter(glm_csv_file, fieldnames=[
                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "R_GLM_test", "R_GLM"])
            writer.writeheader()
    # Load the 900 minute simulated recording
    dataDir = '{}/simulation_data/'.format(ESTIMATOR_DIR)
    analysisDataDir = '/data.nst/lucas/history_dependence/paper/simulation_data/analysis/'
    spiketimes = np.loadtxt('{}spiketimes_constI_5ms.dat'.format(dataDir))
    # Preprocess spiketimes and compute binary counts for current spiking
    spiketimes, counts = glm.preprocess_spiketimes(
        spiketimes, glm_settings)

    # Get the past range for which R should be estimated
    embedding_past_range_set = np.array(
        glm_settings['embedding_past_range_set']).astype(float)
    past_range = embedding_past_range_set[past_range_index]
    # Compute optimized history dependence for given past range
    glm_estimates, glm_estimates_test = glm.compute_estimates_R_cross_validation_new(
        past_range, spiketimes, counts, glm_settings)

    # Save results to glm_benchmarks.csv
    glm.save_glm_estimates_R_to_CSV_Simulation(
        past_range, glm_estimates, glm_estimates_test, glm_settings)
    return EXIT_SUCCESS


# IDEA: Use essentially the benchmark code: fit GLM on one third of the data, but also return the BIC on the full data set. Alternatively, fit on the whole data set. Try 80 past bins for T_index = 35, with counts and with medians.

def main_Simulation_BIC():
    # Get run index for computation on the cluster
    if device == 'cluster':
        past_range_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        past_range_index = 35
    # Load settings
    with open('{}/settings/Simulation_glm.yaml'.format(ESTIMATOR_DIR), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)
    if past_range_index == 0:
        analysis_dir = glm_settings['ANALYSIS_DIR']
        glm_csv_file_name = '{}/glm_estimates_BIC.csv'.format(
            analysis_dir)
        with open(glm_csv_file_name, 'w', newline='') as glm_csv_file:
            writer = csv.DictWriter(glm_csv_file, fieldnames=[
                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size",  "embedding_mode_optimization", "BIC", "R_GLM"])
            writer.writeheader()
    # Load the 900 minute simulated recording
    dataDir = '/home/lucas/research/projects/history_dependence/hdestimator/simulation_data/'
    spiketimes = np.loadtxt('{}spiketimes_constI_5ms.dat'.format(dataDir))
    # Preprocess spiketimes and compute binary counts for current spiking
    spiketimes, counts = glm.preprocess_spiketimes(
        spiketimes, glm_settings)

    # Get the past range for which R should be estimated
    embedding_past_range_set = np.array(
        glm_settings['embedding_past_range_set']).astype(float)
    past_range = embedding_past_range_set[past_range_index]
    # Compute optimized history dependence for given past range

    glm_estimates, BIC = glm.compute_estimates_R_BIC_Simulation(past_range, spiketimes, counts, glm_settings)

    # Save results to glm_benchmarks.csv
    glm.save_glm_estimates_R_BIC_to_CSV_Simulation(
        past_range, glm_estimates, BIC, glm_settings)

    return EXIT_SUCCESS


def main_Experiments():
    rec_length = argv[3]
    # Get run index for computation on the cluster
    if device == 'cluster':
        neuron_index = (int(os.environ['SGE_TASK_ID']) - 1)
    else:
        neuron_index = 0
    # Load settings
    with open('{}/settings/{}_glm.yaml'.format(ESTIMATOR_DIR, recorded_system), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)
    with open('{}/settings/{}_{}.yaml'.format(ESTIMATOR_DIR, recorded_system, setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)
    # Load and preprocess spiketimes and compute binary counts for current spiking
    spiketimes, counts = glm.load_and_preprocess_spiketimes_experiments(
        recorded_system, neuron_index, glm_settings)

    # Get the past range for which R should be estimated
    temporal_depth_bbc, analysis_num_str = glm.get_temporal_depth(
        rec_length, neuron_index, glm_settings, regularization_method = 'bbc')

    embedding_parameters_bbc, analysis_num_str = glm.load_embedding_parameters(
        rec_length, neuron_index, analysis_settings, regularization_method = 'bbc')
    embedding_parameters_bbc = embedding_parameters_bbc[:,
                                                        embedding_parameters_bbc[0] == temporal_depth_bbc]

    # Compute optimized estimate of R for given past range
    glm_estimates, BIC = glm.compute_estimates_R_BIC(
        temporal_depth_bbc, embedding_parameters_bbc, spiketimes, counts, glm_settings)

    # Save results to glm_benchmarks.csv
    glm.save_glm_estimates_R_to_CSV_Experiments(
        temporal_depth_bbc, embedding_parameters_bbc, glm_estimates, BIC, glm_settings, analysis_num_str)

    return EXIT_SUCCESS


if __name__ == "__main__":
    if recorded_system == 'Simulation':
        exit(main_Simulation_BIC())
    else:
        exit(main_Experiments())


############## OLD CODE FOR GLM OPTIMIZATION ON EXPERIMENT ##################
#
# """Indices for computation on the cluster"""
# Tm_index = (int(os.environ['SGE_TASK_ID']) - 1) % 50
# d_index = (int(os.environ['SGE_TASK_ID']) - 1) / 50
#
# """Parameters"""
# #############################################################
# ########### Parameters for the simulated recordings #########
# #############################################################
# # shift time to avoid spiketimes on bin edges, 100 s burnin
# T_0 = 100. - 10**(-4)
# T_f = 54100.  # 10 times 90 min
# T = T_f - T_0  # 90 min recording
# t_bin = 0.005
# mode = 'general'
# ###########################################################
# ### parameters for stepwise upsampling of GLM estimator ###
# ###########################################################
# downsampling = 20
# upsampling = 2.5
# tolerance = 0.0002
# ###########################################################
# ##### parameters for opt d and kappa and fixed Tm   #######
# ###########################################################
# d_list = [10, 20, 40, 60, 80, 100, 120, 150]
# d_emb = d_list[d_index]
# tau_emb = 0.0005
# Tm_0 = 0.01
# Tm_f = 3.
# N_Tm = 50
# Tm_list = np.zeros(N_Tm)
# kappa_Tm = bisect(lambda kappa: np.sum(
#     Tm_0 * np.power(10, np.arange(50) * kappa)) - Tm_f, -1., 10)
# Tm = 0
# for k in range(N_Tm):
#     Tm += Tm_0 * np.power(10, k * kappa_Tm)
#     Tm_list[k] = Tm
# Tm_emb = Tm_list[Tm_index]
# print Tm_emb
# kappa_emb = bisect(lambda kappa: np.sum(
#     tau_emb * np.power(10, np.arange(d_emb) * kappa)) - Tm_emb, -1., 10)
#
# """Preprocess data"""
#
#
# """ Compute M and BIC """
# H_cond, BIC = _H_cond_BIC_GLM(spiketimes, counts, N_bins, d_emb, kappa_emb,
#                               tau_emb, t_bin, T_0, T_f, mode, downsampling, upsampling, tolerance)
# M_GLM = [(H_spike - H_cond) / H_spike]
# np.savetxt('../../Data/Simulated/M_GLM_range%d_d%d.dat' % (Tm_index, d), M_GLM)
# np.savetxt('../../Data/Simulated/BIC_range%d_d%d.dat' % (Tm_index, d), [BIC])
