import numpy as np
import pandas as pd
import yaml


def load_analysis_results_full(recorded_system, rec_length, run_index,  setup, ESTIMATOR_DIR):
    with open('{}/settings/{}_{}.yaml'.format(ESTIMATOR_DIR, recorded_system, setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)
    ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    # Get Rtot, TD and analysis_num from statistics.scv
    statistics_merged_csv_file_name = '{}/statistics_merged.csv'.format(
        ANALYSIS_DIR)
    statistics_pd = pd.read_csv(statistics_merged_csv_file_name)

    index = np.where(statistics_pd['label'] == rec_length +
                     "-" + setup + "-" + str(run_index))[0][0]
    analysis_num_str = str(statistics_pd['#analysis_num'][index])
    for i in range(4 - len(analysis_num_str)):
        analysis_num_str = '0' + analysis_num_str
    R_tot_bbc = statistics_pd['R_tot_bbc'][index]
    T_D_bbc = statistics_pd['T_D_bbc'][index]
    R_tot_shuffling = statistics_pd['R_tot_shuffling'][index]
    T_D_shuffling = statistics_pd['T_D_shuffling'][index]
    # Get T, R(T) plus confidence intervals
    hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    hisdep_pd = pd.read_csv(hisdep_csv_file_name)
    R_bbc = np.array(hisdep_pd['max_R_bbc'])
    R_bbc_CI_lo = np.array(hisdep_pd['max_R_bbc_CI_lo'])
    R_bbc_CI_hi = np.array(hisdep_pd['max_R_bbc_CI_hi'])
    R_shuffling = np.array(hisdep_pd['max_R_shuffling'])
    R_shuffling_CI_lo = np.array(hisdep_pd['max_R_shuffling_CI_lo'])
    R_shuffling_CI_hi = np.array(hisdep_pd['max_R_shuffling_CI_hi'])
    T = np.array(hisdep_pd['#T'])
    return ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, R_tot_shuffling, T_D_shuffling, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi

def load_analysis_results_shuffling(recorded_system, rec_length, run_index, setup, ESTIMATOR_DIR):
    with open('{}/settings/{}_{}.yaml'.format(ESTIMATOR_DIR, recorded_system, setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)
    ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    # Get Rtot, TD and analysis_num from statistics.scv
    statistics_merged_csv_file_name = '{}/statistics_merged.csv'.format(
        ANALYSIS_DIR)
    statistics_pd = pd.read_csv(statistics_merged_csv_file_name)

    index = np.where(statistics_pd['label'] == rec_length +
                     "-" + setup + "-" + str(run_index))[0][0]
    analysis_num_str = str(statistics_pd['#analysis_num'][index])
    for i in range(4 - len(analysis_num_str)):
        analysis_num_str = '0' + analysis_num_str
    R_tot_shuffling = statistics_pd['R_tot_shuffling'][index]
    T_D_shuffling = statistics_pd['T_D_shuffling'][index]
    # Get T, R(T) plus confidence intervals
    hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    hisdep_pd = pd.read_csv(hisdep_csv_file_name)
    R_shuffling = np.array(hisdep_pd['max_R_shuffling'])
    R_shuffling_CI_lo = np.array(hisdep_pd['max_R_shuffling_CI_lo'])
    R_shuffling_CI_hi = np.array(hisdep_pd['max_R_shuffling_CI_hi'])
    T = np.array(hisdep_pd['#T'])
    return R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi


def load_analysis_results_shuffling(recorded_system, rec_length, run_index, setup, ESTIMATOR_DIR):
    with open('{}/settings/{}_{}.yaml'.format(ESTIMATOR_DIR, recorded_system, setup), 'r') as analysis_settings_file:
        analysis_settings = yaml.load(
            analysis_settings_file, Loader=yaml.BaseLoader)
    ANALYSIS_DIR = analysis_settings['ANALYSIS_DIR']
    # Get Rtot, TD and analysis_num from statistics.scv
    statistics_merged_csv_file_name = '{}/statistics_merged.csv'.format(
        ANALYSIS_DIR)
    statistics_pd = pd.read_csv(statistics_merged_csv_file_name)

    index = np.where(statistics_pd['label'] == rec_length +
                     "-" + setup + "-" + str(run_index))[0][0]
    analysis_num_str = str(statistics_pd['#analysis_num'][index])
    for i in range(4 - len(analysis_num_str)):
        analysis_num_str = '0' + analysis_num_str
    R_tot_shuffling = statistics_pd['R_tot_shuffling'][index]
    T_D_shuffling = statistics_pd['T_D_shuffling'][index]
    # Get T, R(T) plus confidence intervals
    hisdep_csv_file_name = '{}/ANALYSIS{}/histdep_data.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    hisdep_pd = pd.read_csv(hisdep_csv_file_name)
    R_shuffling = np.array(hisdep_pd['max_R_shuffling'])
    R_shuffling_CI_lo = np.array(hisdep_pd['max_R_shuffling_CI_lo'])
    R_shuffling_CI_hi = np.array(hisdep_pd['max_R_shuffling_CI_hi'])
    T = np.array(hisdep_pd['#T'])
    return R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi

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
