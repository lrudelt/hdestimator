import argparse
from sys import exit, stderr, argv, path, modules
from os.path import isfile, isdir, realpath, dirname, exists
import h5py
import csv
import ast
import yaml
import numpy as np
from scipy.optimize import minimize, bisect
from scipy.io import loadmat

ESTIMATOR_DIR = dirname(realpath(__file__))
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))

if 'hde_fast_glm' not in modules:
    from hde_fast_glm import*
    import hde_utils as utl

__version__ = "unknown"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def get_temporal_depth(rec_length, neuron_index, glm_settings):
    analysis_dir = glm_settings['ANALYSIS_DIR']
    merged_csv_file_name = '{}/statistics_merged.csv'.format(
        analysis_dir)
    merged_csv_file = open(merged_csv_file_name, 'r')
    # Find the temporal depth for given rec_length and neuron_index
    line_index = 0
    for label in utl.load_from_CSV_file(merged_csv_file, 'label'):
        neuron_index_label = int(label.split("-")[2])
        if neuron_index == neuron_index_label:
            temporal_depth_bbc = float(utl.load_from_CSV_file(
                merged_csv_file, 'T_D_bbc')[line_index])
            analysis_num = int(utl.load_from_CSV_file(
                merged_csv_file, 'analysis_num')[line_index])
            break
        line_index += 1
    merged_csv_file.close()
    analysis_num_str = str(analysis_num)
    for i in range(4 - len(str(analysis_num))):
        analysis_num_str = '0' + analysis_num_str
    return temporal_depth_bbc, temporal_depth_shuffling, analysis_num_str


def get_temporal_depth_Simulation(rec_length, sample_index, analysis_settings):
    analysis_dir = analysis_settings['ANALYSIS_DIR']
    merged_csv_file_name = '{}/statistics_merged.csv'.format(
        analysis_dir)
    merged_csv_file = open(merged_csv_file_name, 'r')
    # Find the temporal depth for given rec_length and sample_index
    line_index = 0
    for label in utl.load_from_CSV_file(merged_csv_file, 'label'):
        sample_index_label = int(label.split("-")[2])
        if sample_index == sample_index_label:
            temporal_depth = float(utl.load_from_CSV_file(
                merged_csv_file, 'T_D_bbc')[line_index])
            temporal_depth_shuffling = float(utl.load_from_CSV_file(
                merged_csv_file, 'T_D_shuffling')[line_index])
            analysis_num = int(utl.load_from_CSV_file(
                merged_csv_file, 'analysis_num')[line_index])
            break
        line_index += 1
    merged_csv_file.close()
    analysis_num_str = str(analysis_num)
    for i in range(4 - len(str(analysis_num))):
        analysis_num_str = '0' + analysis_num_str
    return temporal_depth_bbc, temporal_depth_shuffling, analysis_num_str


def load_embedding_parameters(rec_length, sample_index, analysis_settings):
    analysis_dir = analysis_settings['ANALYSIS_DIR']
    analysis_dir_prefix = 'ANALYSIS'
    merged_csv_file_name = '{}/statistics_merged.csv'.format(
        analysis_dir)
    merged_csv_file = open(merged_csv_file_name, 'r')
    # Find the analysis num for given rec_length and sample_index
    line_index = 0
    for label in utl.load_from_CSV_file(merged_csv_file, 'label'):
        rec_length_label = label.split("-")[0]
        sample_index_label = int(label.split("-")[2])
        if rec_length_label == rec_length and sample_index == sample_index_label:
            analysis_num = int(utl.load_from_CSV_file(
                merged_csv_file, 'analysis_num')[line_index])
            analysis_num_str = str(analysis_num)
            for i in range(4 - len(str(analysis_num))):
                analysis_num_str = '0' + analysis_num_str
            break
        line_index += 1
    merged_csv_file.close()
    # Load the histdep_csv to extract the embeddings for every past range
    histdep_csv_file_name = '{}/{}/histdep_data.csv'.format(
        analysis_dir, analysis_dir_prefix + analysis_num_str)
    histdep_csv_file = open(histdep_csv_file_name, 'r')
    T = utl.load_from_CSV_file(histdep_csv_file, 'T')
    d_bbc = utl.load_from_CSV_file(histdep_csv_file, 'number_of_bins_d_bbc')
    kappa_bbc = utl.load_from_CSV_file(histdep_csv_file, 'scaling_k_bbc')
    tau_bbc = utl.load_from_CSV_file(histdep_csv_file, 'first_bin_size_bbc')
    embedding_parameters_bbc = np.array([T, d_bbc, kappa_bbc, tau_bbc])
    d_shuffling = utl.load_from_CSV_file(
        histdep_csv_file, 'number_of_bins_d_shuffling')
    kappa_shuffling = utl.load_from_CSV_file(
        histdep_csv_file, 'scaling_k_shuffling')
    tau_shuffling = utl.load_from_CSV_file(
        histdep_csv_file, 'first_bin_size_shuffling')
    embedding_parameters_shuffling = np.array([
        T, d_shuffling, kappa_shuffling, tau_shuffling])
    histdep_csv_file.close()
    return embedding_parameters_bbc, embedding_parameters_shuffling, analysis_num_str


def get_embeddings_for_optimization(past_range, d, max_first_bin_size):
    uniform_bin_size = past_range / d
    if d == 1:
        tau = uniform_bin_size
        kappa = 0.0
    else:
        if uniform_bin_size <= max_first_bin_size:
            # if uniform bins are small enough, than use uniform embedding
            tau = uniform_bin_size
            kappa = 0.0
        else:
             # If the bin size with uniform bins is > max_first_bin_size, then choose exponential embedding such that first bin size is equal to minimum first bin size
            tau = max_first_bin_size
            kappa = bisect(lambda k: np.sum(
                max_first_bin_size * np.power(10, np.arange(d) * k)) - past_range, -1., 10)
    return kappa, tau


def preprocess_spiketimes(spiketimes, glm_settings):
    T_0 = float(glm_settings['burning_in_time'])
    t_bin = float(glm_settings['embedding_step_size'])
    T_f = T_0 + float(glm_settings['total_recording_length'])
    # Shift spiketimes to ignore spikes from the burn in period, because analysis starts at time 0
    spiketimes = spiketimes - T_0
    # computes binary spike counts to represent current spiking
    counts = counts_C(spiketimes, t_bin, T_0, T_f, 'binary')
    return spiketimes, counts


def load_and_preprocess_spiketimes_experiments(recorded_system, neuron_index, glm_settings):
    if recorded_system == 'EC':
        dataDir = '/data.nst/lucas/history_dependence/paper/EC_data/'
        validNeurons = np.load(
            '{}validNeurons.npy'.format(dataDir)).astype(int)
        neuron = validNeurons[neuron_index]
        sptimes_raw = loadmat('{}spks/ec014.277.spike_ch.mat'.format(dataDir))
        sample_rate = 20000.
        sptimes = sptimes_raw['sptimes'][0] / sample_rate
        spiketimes = sptimes[neuron].flatten()
    if recorded_system == 'Retina':
        dataDir = '/data.nst/lucas/history_dependence/paper/retina_data/'
        validNeurons = np.load(
            '{}validNeurons.npy'.format(dataDir)).astype(int)
        neuron = validNeurons[neuron_index]
        spiketimes = np.loadtxt(
            '{}spks/spiketimes_neuron{}.dat'.format(dataDir, neuron))
    if recorded_system == 'Culture':
        dataDir = '/data.nst/lucas/history_dependence/paper/culture_data/'
        validNeurons = np.load(
            '{}validNeurons.npy'.format(dataDir)).astype(int)
        neuron = validNeurons[neuron_index]
        spiketimes = np.loadtxt(
            '{}spks/spiketimes_neuron{}.dat'.format(dataDir, neuron))
        sample_rate = 24038.46169
        spiketimes = spiketimes / sample_rate

    t_bin = float(glm_settings['embedding_step_size'])
    # Offset starting time of the analysis such that at least 5 seconds of spiking history are observed
    T_0 = spiketimes[1] + 5.
    T_f_recording = spiketimes[-1] - 2.
    T_f_max = T_0 + float(glm_settings['total_recording_length'])
    T_f = np.amin([T_f_recording, T_f_max])
    # Shift spiketimes by starting time T_0, because analysis starts at time 0
    spiketimes = spiketimes - T_0
    # computes binary spike counts to represent current spiking
    counts = counts_C(spiketimes, t_bin, T_0, T_f, 'binary')
    return spiketimes, counts


def save_glm_benchmark_to_CSV(glm_benchmark, embedding_parameters, analysis_settings, analysis_num_str, regularization_method='bbc'):
    analysis_dir = analysis_settings['ANALYSIS_DIR']
    analysis_dir = analysis_dir + '/ANALYSIS' + analysis_num_str
    glm_csv_file_name = '{}/glm_benchmark_{}.csv'.format(
        analysis_dir, regularization_method)
    with open(glm_csv_file_name, 'w', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "R_GLM"])
        writer.writeheader()
        for i, T in enumerate(embedding_parameters[0]):
            writer.writerow(
                {"T": T, "number_of_bins_d": int(embedding_parameters[1][i]), "scaling_kappa": embedding_parameters[2][i], "first_bin_size": embedding_parameters[3][i], "R_GLM": glm_benchmark[i]})
    return EXIT_SUCCESS


def save_glm_estimates_R_to_CSV_Simulation(past_range, glm_estimates, glm_estimates_test, glm_settings):
    analysis_dir = glm_settings['ANALYSIS_DIR']
    glm_csv_file_name = '{}/glm_estimates_cross_validated_new.csv'.format(
        analysis_dir)
    with open(glm_csv_file_name, 'a+', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "R_GLM_test", "R_GLM"])
        for i, d in enumerate(glm_settings['embedding_number_of_bins_set']):
            d = int(d)
            max_first_bin_size = float(glm_settings['max_first_bin_size'])
            kappa, tau = get_embeddings_for_optimization(
                past_range, d, max_first_bin_size)
            writer.writerow(
                {"T": past_range, "number_of_bins_d": d, "scaling_kappa": kappa, "first_bin_size": tau, "R_GLM_test": glm_estimates_test[i], "R_GLM": glm_estimates[i]})
    return EXIT_SUCCESS


def save_glm_estimates_R_to_CSV_Experiments(past_range, glm_estimates, BIC, glm_settings, analysis_num_str):
    analysis_dir = glm_settings['ANALYSIS_DIR']
    analysis_dir = analysis_dir + '/ANALYSIS' + analysis_num_str
    glm_csv_file_name = '{}/glm_estimates_BIC.csv'.format(
        analysis_dir)
    with open(glm_csv_file_name, 'w', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "BIC", "R_GLM"])
        writer.writeheader()
        for i, d in enumerate(glm_settings['embedding_number_of_bins_set']):
            d = int(d)
            max_first_bin_size = float(glm_settings['max_first_bin_size'])
            kappa, tau = get_embeddings_for_optimization(
                past_range, d, max_first_bin_size)
            writer.writerow(
                {"T": past_range, "number_of_bins_d": d, "scaling_kappa": kappa, "first_bin_size": tau, "BIC": BIC[i], "R_GLM": glm_estimates[i]})
    return EXIT_SUCCESS


def fit_GLM_params(counts, past, d, N_bins):
    mu_0 = 1.
    h_0 = np.zeros(d)
    res = minimize(lambda param: -L_B_past(counts, past, d, N_bins, param[1:], param[0]), np.append([mu_0], h_0), method='Newton-CG', jac=lambda param: -jac_L_B_past(
        counts, past, d, N_bins, param[1:], param[0]), hess=lambda param: -hess_L_B_past(counts, past, d, N_bins, param[1:], param[0]))
    mu = res.x[0]
    h = res.x[1:]
    return mu, h


def compute_R_GLM(counts, past, d, N, mu, h):
    P_spike = np.sum(counts) / float(N)
    H_spike = -P_spike * np.log(P_spike) - (1 - P_spike) * np.log(1 - P_spike)
    R_GLM = 1 - H_cond_B_past(counts, past, d, N, mu, h) / H_spike
    return R_GLM


def compute_BIC_GLM(counts, past, d, N, mu, h):
    BIC = -2 * L_B_past(counts, past, d, N, h, mu) + (d + 1) * np.log(N)
    return BIC


def compute_benchmark_R(embedding_parameters, spiketimes, counts, glm_settings, regularization_method='bbc'):
    t_bin = float(glm_settings['embedding_step_size'])
    embedding_mode = glm_settings['embedding_mode_benchmark']
    # Number of total data points
    N = len(counts)
    # Number of training data points and indices for training data set
    N_training = int(N / 3)  # train on one third of the data
    np.random.seed(42)
    training_indices = np.random.choice(N, N_training, replace=False)
    counts_training = counts[training_indices]

    R_GLM = []
    for i, T in enumerate(embedding_parameters[0]):
        # embedding parameters
        d = int(embedding_parameters[1][i])
        kappa = embedding_parameters[2][i]
        tau = embedding_parameters[3][i]
        # apply past embedding
        past = past_activity(spiketimes, d, kappa, tau,
                             t_bin, N, embedding_mode)
        # downsample to obtain smaller training data set to speed up fitting
        past_training = downsample_past_activity(
            past, training_indices, N, d)
        # fit GLM parameters
        mu, h = fit_GLM_params(
            counts_training, past_training, d, N_training)
        # estimate history dependence for fitted GLM parameters
        R_GLM += [compute_R_GLM(counts, past, d, N, mu, h)]

    return R_GLM

# Compute estimates of R as well as on a test data set for given past range and a set of embedding dimensions d


def compute_estimates_R_cross_validation_new(past_range, spiketimes, counts_total, glm_settings):
    # Load embedding parameters for optimization
    t_bin = float(glm_settings['embedding_step_size'])
    embedding_mode = glm_settings['embedding_mode_optimization']
    embedding_number_of_bins_set = np.array(
        glm_settings['embedding_number_of_bins_set']).astype(int)
    max_first_bin_size = float(glm_settings['max_first_bin_size'])
    # Number of total data points
    N = len(counts_total)
    # test on one third of the data
    N_test = int(
        N / 3)
    N_training = N - N_test
    # Random, non-overlapping indices for train, test and evaluation data sets
    np.random.seed(42)
    permuted_indices = np.random.permutation(N)
    training_indices = permuted_indices[:N_training]
    test_indices = permuted_indices[N_training:N_training + N_test]

    R_GLM_test = []
    R_GLM = []
    for d in embedding_number_of_bins_set:
        # get remaining embedding parameters such that the embedding has a certain minimum resolution (set by max_first_bin_size)
        kappa, tau = get_embeddings_for_optimization(
            past_range, d, max_first_bin_size)

        # apply past embedding
        past_total = past_activity(spiketimes, d, kappa, tau,
                                   t_bin, N, embedding_mode)
        # Fit GLM parameters on training data set
        past = downsample_past_activity(
            past_total, training_indices, N_training, d)
        counts = counts_total[training_indices]
        mu, h = fit_GLM_params(
            counts, past, d, N_training)
        R_GLM += [compute_R_GLM(counts,
                                past, d, N_training, mu, h)]
        # Evaluate GLM estimates of R for fitted GLM parameters on test data set
        past = downsample_past_activity(
            past_total, test_indices, N_test, d)
        counts = counts_total[test_indices]
        R_GLM_test += [compute_R_GLM(counts, past, d, N_test, mu, h)]

    return R_GLM, R_GLM_test


def compute_estimates_R_cross_validation(past_range, spiketimes, counts_total, glm_settings):
    # Load embedding parameters for optimization
    t_bin = float(glm_settings['embedding_step_size'])
    embedding_mode = glm_settings['embedding_mode_optimization']
    embedding_number_of_bins_set = np.array(
        glm_settings['embedding_number_of_bins_set']).astype(int)
    max_first_bin_size = float(glm_settings['max_first_bin_size'])
    # Number of total data points
    N = len(counts_total)
    # train, test and evaluate on one third of the data
    N_training = N_test = N_evaluation = int(
        N / 3)
    # Random, non-overlapping indices for train, test and evaluation data sets
    np.random.seed(42)
    permuted_indices = np.random.permutation(N)
    training_indices = permuted_indices[:N_training]
    test_indices = permuted_indices[N_training:N_training + N_test]
    evaluation_indices = permuted_indices[N_training + N_test:N]

    R_GLM_test = []
    R_GLM_evaluation = []
    for d in embedding_number_of_bins_set:
        # get remaining embedding parameters such that the embedding has a certain minimum resolution (set by max_first_bin_size)
        kappa, tau = get_embeddings_for_optimization(
            past_range, d, max_first_bin_size)

        # apply past embedding
        past_total = past_activity(spiketimes, d, kappa, tau,
                                   t_bin, N, embedding_mode)
        # Fit GLM parameters on training data set
        past = downsample_past_activity(
            past_total, training_indices, N_training, d)
        counts = counts_total[training_indices]
        mu, h = fit_GLM_params(
            counts, past, d, N_training)
        # Evaluate GLM estimates of R for fitted GLM parameters on test data set
        past = downsample_past_activity(
            past_total, test_indices, N_test, d)
        counts = counts_total[test_indices]
        R_GLM_test += [compute_R_GLM(counts, past, d, N_test, mu, h)]
        # Evaluate GLM estimates of R for fitted GLM parameters on test data set
        past = downsample_past_activity(
            past_total, evaluation_indices, N_evaluation, d)
        counts = counts_total[evaluation_indices]
        R_GLM_evaluation += [compute_R_GLM(counts,
                                           past, d, N_evaluation, mu, h)]

    return R_GLM_test, R_GLM_evaluation

# Compute estimates of R and the Bayesian information criterion (BIC) for given past range and a set of embedding dimensions d


def compute_estimates_R_BIC(past_range, spiketimes, counts, glm_settings):
    # Load embedding parameters for optimization
    t_bin = float(glm_settings['embedding_step_size'])
    embedding_mode = glm_settings['embedding_mode_optimization']
    embedding_number_of_bins_set = np.array(
        glm_settings['embedding_number_of_bins_set']).astype(int)
    max_first_bin_size = float(glm_settings['max_first_bin_size'])
    # Number of total data points
    N = len(counts)
    R_GLM = []
    BIC = []
    for d in embedding_number_of_bins_set:
        # get remaining embedding parameters such that the embedding has a certain minimum resolution (set by max_first_bin_size)
        kappa, tau = get_embeddings_for_optimization(
            past_range, d, max_first_bin_size)
        # apply past embedding
        past = past_activity(spiketimes, d, kappa, tau,
                             t_bin, N, embedding_mode)
        # Fit GLM parameters on the whole data set
        mu, h = fit_GLM_params(counts, past, d, N)
        # Evaluate GLM estimate of R for fitted GLM parameters on the whole data set
        R_GLM += [compute_R_GLM(counts, past, d, N, mu, h)]
        # Compute BIC for fitted GLM parameters on whole data set
        BIC += [compute_BIC_GLM(counts, past, d, N, mu, h)]
    return R_GLM, BIC


##############3 OLD STUFF #################

    # Estimate R based on fitted GLM parameters

    # H_cond, BIC = _H_cond_BIC_GLM(spiketimes, counts, N_bins, d_emb, kappa_emb,
    #                               tau_emb, t_bin, T_0, T_f, mode, downsampling, upsampling, tolerance)
    #
    # # Optimize parameters on the first "training" third of the recording, then output likelihood on the second "test" third to choose emb parameters and compute histdep estimate on the last "evaluation" third of the recording
    #
    #
    # def _R_GLM_cross_validated(spiketimes, d, kappa, tau, mode, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out=False):
    #     if downsampling_ratio == 1:
    #         N_sample_list = [N_bins]
    #     else:
    #         N_sample_list = [int(N_bins * downsampling_ratio), N_bins]
    #     h_0 = np.zeros(d)
    #     mu_0 = 1.
    #     for N_sample in N_sample_list:
    #         print N_sample
    #         indices_inference = np.arange(N_sample).astype(int)
    #         counts_inference = counts[indices_inference]
    #         indices_inference = np.append(indices_inference, 0)
    #         y_t = past_activity(
    #             spiketimes, indices_inference, d, N_sample, kappa, tau, t_bin, T_0, T_f, mode)
    #         # Learn the parameters on a smaller sample for speed
    #         res = minimize(lambda param: -L_B_past(counts_inference, y_t, d, N_sample, param[1:], param[0]), np.append([mu_0], h_0), method='Newton-CG', jac=lambda param: -jac_L_B_past(
    #             counts_inference, y_t, d, N_sample, param[1:], param[0]), hess=lambda param: -hess_L_B_past(counts_inference, y_t, d, N_sample, param[1:], param[0]))
    #         mu = res.x[0]
    #         h = res.x[1:]
    #         mu_0 = mu
    #         h_0 = h
    #     # Compute conditional entropy with respect to the sample
    #     H_cond = H_cond_B_past(
    #         counts_inference, y_t, d, N_sample, h, mu)
    #     BIC = -2 * L_B_past(counts_inference, y_t, d, N_sample, h, mu) + (
    #         d + 1) * np.log(N_sample)  # Compute BIC with respect to the sample
    #     if params_out == True:
    #         return H_cond, BIC, mu, h
    #     else:
    #         return H_cond, BIC
    #
    # # Do not separate data sets but compute the BIC to optimized embedding.
    #
    #
    # def _H_cond_GLM_BIC(spiketimes, d, kappa, tau, mode, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out=False):
    #     if downsampling_ratio == 1:
    #         N_sample_list = [N_bins]
    #     else:
    #         N_sample_list = [int(N_bins * downsampling_ratio), N_bins]
    #     h_0 = np.zeros(d)
    #     mu_0 = 1.
    #     for N_sample in N_sample_list:
    #         print N_sample
    #         indices_inference = np.arange(N_sample).astype(int)
    #         counts_inference = counts[indices_inference]
    #         indices_inference = np.append(indices_inference, 0)
    #         y_t = past_activity(
    #             spiketimes, indices_inference, d, N_sample, kappa, tau, t_bin, T_0, T_f, mode)
    #         # Learn the parameters on a smaller sample for speed
    #         res = minimize(lambda param: -L_B_past(counts_inference, y_t, d, N_sample, param[1:], param[0]), np.append([mu_0], h_0), method='Newton-CG', jac=lambda param: -jac_L_B_past(
    #             counts_inference, y_t, d, N_sample, param[1:], param[0]), hess=lambda param: -hess_L_B_past(counts_inference, y_t, d, N_sample, param[1:], param[0]))
    #         mu = res.x[0]
    #         h = res.x[1:]
    #         mu_0 = mu
    #         h_0 = h
    #     # Compute conditional entropy with respect to the sample
    #     H_cond = H_cond_B_past(
    #         counts_inference, y_t, d, N_sample, h, mu)
    #     BIC = -2 * L_B_past(counts_inference, y_t, d, N_sample, h, mu) + (
    #         d + 1) * np.log(N_sample)  # Compute BIC with respect to the sample
    #     if params_out == True:
    #         return H_cond, BIC, mu, h
    #     else:
    #         return H_cond, BIC
