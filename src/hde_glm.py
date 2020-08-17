import argparse
from sys import exit, stderr, argv, path
from os.path import isfile, isdir, realpath, dirname, exists
import h5py
import ast
import yaml
import numpy as np
from scipy.optimize import minimize
from _version import __version__

ESTIMATOR_DIR = dirname(realpath(__file__))
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))

if 'hde_fast_glm' not in sys.modules:
    from hde_fast_glm as import*

__version__ = "unknown"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# THis is how you load yaml settings
# TODO: get list of past_ranges from the settings file, as well as the GLM parameters
with open('{}/settings/default.yaml'.format(ESTIMATOR_DIR), 'r') as default_settings_file:
    settings = yaml.load(default_settings_file, Loader=yaml.BaseLoader)


# Optimize parameters on the first "training" third of the recording, then output likelihood on the second "test" third to choose emb parameters and compute histdep estimate on the last "evaluation" third of the recording


def _H_cond_GLM_cross_validated(spiketimes, d, kappa, tau, mode, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out=False):
    if downsampling_ratio == 1:
        N_sample_list = [N_bins]
    else:
        N_sample_list = [int(N_bins * downsampling_ratio), N_bins]
    h_0 = np.zeros(d)
    mu_0 = 1.
    for N_sample in N_sample_list:
        print N_sample
        indices_inference = np.arange(N_sample).astype(int)
        counts_inference = counts[indices_inference]
        indices_inference = np.append(indices_inference, 0)
        y_t = past_activity(
            spiketimes, indices_inference, d, N_sample, kappa, tau, t_bin, T_0, T_f, mode)
        # Learn the parameters on a smaller sample for speed
        res = minimize(lambda param: -L_B_past(counts_inference, y_t, d, N_sample, param[1:], param[0]), np.append([mu_0], h_0), method='Newton-CG', jac=lambda param: -jac_L_B_past(
            counts_inference, y_t, d, N_sample, param[1:], param[0]), hess=lambda param: -hess_L_B_past(counts_inference, y_t, d, N_sample, param[1:], param[0]))
        mu = res.x[0]
        h = res.x[1:]
        mu_0 = mu
        h_0 = h
    # Compute conditional entropy with respect to the sample
    H_cond = H_cond_B_past(
        counts_inference, y_t, d, N_sample, h, mu)
    BIC = -2 * L_B_past(counts_inference, y_t, d, N_sample, h, mu) + (
        d + 1) * np.log(N_sample)  # Compute BIC with respect to the sample
    if params_out == True:
        return H_cond, BIC, mu, h
    else:
        return H_cond, BIC

# Do not separate data sets but compute the BIC to optimized embedding.


def _H_cond_GLM_cross_BIC(spiketimes, d, kappa, tau, mode, counts, N_bins, t_bin, T_0, T_f, downsampling_ratio, params_out=False):
    if downsampling_ratio == 1:
        N_sample_list = [N_bins]
    else:
        N_sample_list = [int(N_bins * downsampling_ratio), N_bins]
    h_0 = np.zeros(d)
    mu_0 = 1.
    for N_sample in N_sample_list:
        print N_sample
        indices_inference = np.arange(N_sample).astype(int)
        counts_inference = counts[indices_inference]
        indices_inference = np.append(indices_inference, 0)
        y_t = past_activity(
            spiketimes, indices_inference, d, N_sample, kappa, tau, t_bin, T_0, T_f, mode)
        # Learn the parameters on a smaller sample for speed
        res = minimize(lambda param: -L_B_past(counts_inference, y_t, d, N_sample, param[1:], param[0]), np.append([mu_0], h_0), method='Newton-CG', jac=lambda param: -jac_L_B_past(
            counts_inference, y_t, d, N_sample, param[1:], param[0]), hess=lambda param: -hess_L_B_past(counts_inference, y_t, d, N_sample, param[1:], param[0]))
        mu = res.x[0]
        h = res.x[1:]
        mu_0 = mu
        h_0 = h
    # Compute conditional entropy with respect to the sample
    H_cond = H_cond_B_past(
        counts_inference, y_t, d, N_sample, h, mu)
    BIC = -2 * L_B_past(counts_inference, y_t, d, N_sample, h, mu) + (
        d + 1) * np.log(N_sample)  # Compute BIC with respect to the sample
    if params_out == True:
        return H_cond, BIC, mu, h
    else:
        return H_cond, BIC


def main():
    """
    Parse arguments and settings and then run selected tasks.
    """

    # definitions
    defined_tasks = ["history-dependence",
                     "confidence-intervals",
                     # "permutation-test",
                     "auto-mi",
                     "csv-files",
                     "plots",
                     "full-analysis"]

    defined_estimation_methods = ['bbc', 'shuffling', 'all']

    # get task and target (parse arguments and check for validity)
    task, spike_times, spike_times_optimization, spike_times_validation, \
        analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, analysis_num, \
        settings = parse_arguments(defined_tasks,
                                   defined_estimation_methods)

    if settings['estimation_method'] == 'all':
        estimation_methods = ['bbc', 'shuffling']
    else:
        estimation_methods = [settings['estimation_method']]

    # now perform tasks as specified by the parsed arguments

    for estimation_method in estimation_methods:
        settings['estimation_method'] = estimation_method

        if task == "history-dependence" or task == "full-analysis":
            do_main_analysis(spike_times, spike_times_optimization, spike_times_validation,
                             analysis_file, settings)

        if task == "confidence-intervals" or task == "full-analysis":
            compute_CIs(spike_times_validation, analysis_file, settings)

        # if task == "permutation-test" or task == "full-analysis":
        #     perform_permutation_test(analysis_file, settings)

    if task == "auto-mi" or task == "full-analysis":
        analyse_auto_MI(spike_times, analysis_file, settings)

    if task == "csv-files" or task == "full-analysis":
        create_CSV_files(analysis_file,
                         csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file,
                         analysis_num, settings)

    if task == "plots" or task == "full-analysis":
        produce_plots(spike_times,
                      csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file,
                      settings)

    for f in [analysis_file,
              csv_stats_file,
              csv_histdep_data_file,
              csv_auto_MI_data_file]:
        if not f == None:
            f.close()

    return EXIT_SUCCESS


if __name__ == "__main__":
    exit(main())
