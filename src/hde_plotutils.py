import numpy as np
import pandas as pd
import yaml

def load_analysis_results(recorded_system, rec_length, run_index, setup, ESTIMATOR_DIR, regularization_method = 'bbc'):
    with open('{}/settings/{}_{}.yaml'.format(ESTIMATOR_DIR, recorded_system, setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)
    ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    # Get Rtot, TD and analysis_num from statistics.scv
    statistics_merged_csv_file_name = '{}/statistics_merged.csv'.format(
        ANALYSIS_DIR)
    statistics_pd = pd.read_csv(statistics_merged_csv_file_name)
    index = np.where(statistics_pd['label'] == rec_length + "-" + setup + "-" + str(run_index))[0][0]
    analysis_num_str = str(statistics_pd['#analysis_num'][index])
    for i in range(4 - len(analysis_num_str)):
        analysis_num_str = '0' + analysis_num_str
    R_tot = statistics_pd['R_tot_{}'.format(regularization_method)][index]
    T_D = statistics_pd['T_D_{}'.format(regularization_method)][index]
    # Get T, R(T) plus confidence intervals
    hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    hisdep_pd = pd.read_csv(hisdep_csv_file_name)
    R = np.array(hisdep_pd['max_R_{}'.format(regularization_method)])
    R_CI_lo = np.array(hisdep_pd['max_R_{}_CI_lo'.format(regularization_method)])
    R_CI_hi = np.array(hisdep_pd['max_R_{}_CI_hi'.format(regularization_method)])
    T = np.array(hisdep_pd['#T'])
    if regularization_method == 'bbc':
        return ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi
    else:
        return R_tot, T_D, T, R, R_CI_lo, R_CI_hi

def load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str):
    glm_csv_file_name = '{}/ANALYSIS{}/glm_estimates_BIC.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    glm_pd = pd.read_csv(glm_csv_file_name)
    index = np.argmin(glm_pd['BIC'])
    R_tot_glm = glm_pd['R_GLM'][index]
    return R_tot_glm

def get_CI_median(samples):
    N_samples = len(samples)
    median = np.median(samples)
    median_samples = np.sort(np.median(np.random.choice(
        samples, size=(10000, N_samples)), axis=1))
    CI_lo = median_samples[249]
    CI_hi = median_samples[9749]
    return CI_lo, CI_hi


def get_CI_mean(samples):
    N_samples = len(samples)
    mean = np.mean(samples)
    mean_samples = np.sort(np.mean(np.random.choice(
        samples, size=(10000, N_samples)), axis=1))
    CI_lo = mean_samples[249]
    CI_hi = mean_samples[9749]
    return CI_lo, CI_hi


def get_R_tot(T, R, R_CI_lo):
    R_max = np.amax(R)
    std_R_max = (R_max - R_CI_lo[R == np.amax(R)][np.nonzero(R_max - R_CI_lo[R == np.amax(R)])][0])/2
    T_D = T[R > R_max - std_R_max][0]
    T_max_valid = T[R > R_max - std_R_max][-1]
    T_D_index = np.where(T == T_D)[0][0]
    max_valid_index = np.where(T == T_max_valid)[0][0]+1
    R_tot = np.mean(R[T_D_index:max_valid_index])
    return R_tot, T_D_index, max_valid_index


def get_temporal_depth_and_R_tot(T, R, min_number_of_neighbors_for_avg = 2, minimum_offset = 0.01):
    n = min_number_of_neighbors_for_avg
    running_avg = np.array([np.mean(R[i-n:i+n+1]) for i in np.arange(n,len(T)-n)])
    running_std = np.array([np.std(R[i-n:i+n+1]) for i in np.arange(n,len(T)-n)])
    max_running_avg = np.amax(running_avg)
    eff_max_running_avg = running_avg[running_avg>=max_running_avg][0]
    index = np.where(running_avg==eff_max_running_avg)[0][0]
    max_running_std = running_std[index]
    max_running_avg = running_avg[index]
    index = index+n
    # check saturated estimates for high T and stop if saturated
    i = 1
    while R[-i] < max_running_avg - 2 * max_running_std :
            i+=1
    max_valid_index = len(R) - i + 1
    # update the mean and std for the new set of viable estimates
    max_running_avg = np.mean(R[index-n:max_valid_index])
    # check saturated estimates low T and stop when saturated
    # i = 0
    # offset = np.amax([minimum_offset*max_running_avg, 2 * max_running_std])
    # while R[i] < max_running_avg - offset:
    #     i+=1
    # T_D_index = i

    i = 0
    while R[i] < np.amax([np.mean(R[i:max_valid_index])- np.std(R[i:max_valid_index]), max_running_avg*.95]) :
        i+=1
    T_D_index = i

    # max_running_avg = np.mean(R[T_D_index-n:max_valid_index])
    # max_running_std = np.std(R[T_D_index-n:max_valid_index])
    #
    # i = 1
    # while R[-i] < max_running_avg - 2 * max_running_std:
    #         i+=1
    # max_valid_index = len(R) - i + 1
    #
    # max_running_avg = np.mean(R[T_D_index-n:max_valid_index])
    # max_running_std = np.std(R[T_D_index-n:max_valid_index])
    # # check saturated estimates low T and stop when saturated
    # i = 0
    # while R[i] < max_running_avg - 2 * max_running_std:
    #     i+=1
    # T_D_index = i

    # j = 0
    # while max_running_avg > running_avg[-j-1]+running_std[-j-1]:
    #     j += 1
    # if j == 0:
    #     max_valid_index = len(T)
    # else:
    #     max_valid_index = len(T) - 2 - j
    # k = 3
    # possible_R = []
    # possible_R_indices = []
    # # goes through values that are in the range of valid estimates, but at most k values before the last past range (to allow a meaningful average)
    # T_D_index = 0
    # for i, R_val in enumerate(R[:np.amin([len(T)-k,max_valid_index])]):
    #     if i > 0:
    #         offset = np.amax([np.std(R[i:max_valid_index]),minimum_offset])
    #         if R_val > np.mean(R[i:max_valid_index]) - minimum_offset:
    #             T_D_index = i
    #             break
    # for i, R_val in enumerate(R[:np.amin([len(T)-k,max_valid_index])]):
    #     if i > 0:
    #         if R_val < np.mean(R[i:max_valid_index]):
    #             if R_val-np.std(R[i:max_valid_index]) > np.amax(R[:i]):
    #                 possible_R += [R_val]
    #                 possible_R_indices += [i]
    # T_D_index = possible_R_indices[-1]
    T_D = T[T_D_index]
    R_tot = np.mean(R[T_D_index:max_valid_index])
    R_tot_std = np.std(R[T_D_index:max_valid_index])
    return T_D, R_tot, R_tot_std, T_D_index, max_valid_index
