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
import pandas as pd

ESTIMATOR_DIR = '{}/..'.format(dirname(realpath(__file__)))
# ESTIMATOR_DIR = '/home/lucas/research/projects/history_dependence/hdestimator'
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))

if 'hde_fast_glm' not in modules:
    from hde_fast_glm import*
    import hde_utils as utl

__version__ = "unknown"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

"""Run parameters"""
# recorded_system = argv[1]
recorded_system = 'Simulation'


def main_Simulation():
    with open('{}/settings/Simulation_glm.yaml'.format(ESTIMATOR_DIR), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)
    analysis_dir = glm_settings['ANALYSIS_DIR']
    glm_csv_file_name = '{}/glm_estimates_cross_validated.csv'.format(
        analysis_dir)
    glm_merged_csv_file_name = '{}/glm_opt_estimates_cross_validated.csv'.format(
        analysis_dir)
    glm_pd = pd.read_csv(glm_csv_file_name)
    glm_values = glm_pd.values
    T_list = glm_values[:, 0]
    with open(glm_merged_csv_file_name, 'w', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "R_GLM_opt"])
        writer.writeheader()
        for T in np.unique(T_list):
            indices = np.where(T_list == T)[0]
            R_GLM_test_list = glm_values[indices, 4]
            opt_index = indices[np.argmax(R_GLM_test_list)]
            opt_values = glm_values[opt_index]
            writer.writerow(
                {"T": T, "number_of_bins_d": opt_values[1], "scaling_kappa": opt_values[2], "first_bin_size": opt_values[3], "R_GLM_opt": opt_values[5]})
    return EXIT_SUCCESS


def main_Experiments():
    rec_length = argv[2]
    with open('{}/settings/Simulation_glm.yaml'.format(ESTIMATOR_DIR), 'r') as glm_settings_file:
        glm_settings = yaml.load(glm_settings_file, Loader=yaml.BaseLoader)
    analysis_dir = glm_settings['ANALYSIS_DIR']
    glm_csv_file_name = '{}/glm_estimates_cross_validated.csv'.format(
        analysis_dir)
    glm_merged_csv_file_name = '{}/glm_opt_estimates_cross_validated.csv'.format(
        analysis_dir)
    glm_pd = pd.read_csv(glm_csv_file_name)
    glm_values = glm_pd.values
    with open(glm_merged_csv_file_name, 'w', newline='') as glm_csv_file:
        writer = csv.DictWriter(glm_csv_file, fieldnames=[
                                "T", "number_of_bins_d", "scaling_kappa", "first_bin_size", "R_GLM_opt"])
        writer.writeheader()
        # TODO: there are certainly things wrong with the queries, but when you fixed this should be fine
        for T in np.array(glm_settings['embedding_past_range_set']).astype(float):
            T_list = glm_values[:, 0]
            indices = np.where(T_list == T)
            R_GLM_test_list = glm_values[indices, 4]
            opt_index = indices[np.argmax(R_GLM_test_list)]
            opt_values = glm_values[opt_index]
            writer.writerow(
                {"T": T, "number_of_bins_d": opt_values[1], "scaling_kappa": opt_values[2], "first_bin_size": opt_values[3], "R_GLM_opt": opt_values[5]})

    return EXIT_SUCCESS


if __name__ == "__main__":
    if recorded_system == 'Simulation':
        exit(main_Simulation())
    else:
        exit(main_Experiments())
