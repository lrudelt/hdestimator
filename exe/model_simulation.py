import numpy as np
import random

codedirectory = '/home/lucas/research/projects/history_dependence/hdestimator'


def _P(V, V_T, DeltaV, lambda_0, t_step):
    x = lambda_0*np.exp((V-V_T)/DeltaV)*t_step
    return x/(1+x)


def _threshold_adaptation(past_spikes, settings):
    a_thresh = settings["a_thresh"]
    beta_thresh = settings["beta_thresh"]
    T_thresh = settings["T_thresh"]
    threshold_adaptation = 0
    for s in past_spikes:
        if s < T_thresh:
            threshold_adaptation += a_thresh
        else:
            threshold_adaptation += np.power(s/T_thresh, -beta_thresh)*a_thresh
    return threshold_adaptation


def _threshold_adaptation_nospike(past_spikes, settings, s_bin_edge):
    a_thresh = settings["a_thresh"]
    beta_thresh = settings["beta_thresh"]
    T_thresh = settings["T_thresh"]
    threshold_adaptation = 0
    for s in past_spikes:
        if s > s_bin_edge:
            if s < T_thresh:
                threshold_adaptation += a_thresh
            else:
                threshold_adaptation += np.power(s/T_thresh, -beta_thresh)*a_thresh
    return threshold_adaptation


def _integrate_membrane_potential(V, ref_count, settings, t_step):
    tau = settings["tau"]
    E_L = settings["E_L"]
    R = settings["R"]
    I_0 = settings["I_0"]
    V_R = settings["V_R"]
    V_T = settings["V_T"]
    # Refractoriness
    if ref_count > 0:
        V = V_R
    # Dynamics outside the refractory period
    else:
        V += -t_step/tau*(V-E_L)+R*I_0
    return V


def _compute_nospike_prob(V_nospike,  P_nospike, counts, past_spikes, step_index, bin_index, ref_count, number_steps_per_bin, t_step, settings):
    tau = settings["tau"]
    E_L = settings["E_L"]
    R = settings["R"]
    I_0 = settings["I_0"]
    V_R = settings["V_R"]
    V_T = settings["V_T"]
    DeltaV = settings["DeltaV"]
    lambda_0 = settings["lambda_0"]
    T_ref = settings["T_ref"]
    # distance to past bin edge has offset T_ref like the stored past spikes
    s_bin_edge = (step_index % number_steps_per_bin) * t_step - T_ref
    V_T_dyn_nospike = V_T+_threshold_adaptation_nospike(past_spikes, settings, s_bin_edge)
    if ref_count > 0:
        # If refractoriness is caused by a spike within the same bin, then the nospike dynamics should ignore it. Otherwise the refractoriness is also considered for bin probability
        if counts[bin_index] > 0:
            # simulate membrane dynamics without spikes in the current time bin
            V_nospike += -t_step/tau*(V_nospike-E_L)+R*I_0
            # Compute probabity for no spike in the time bin, given only information of past spikes before the bin
            P_nospike[bin_index] = P_nospike[bin_index]*(1-_P(V_nospike, V_T_dyn_nospike, DeltaV, lambda_0, t_step))
        else:
            V_nospike = V_R
    else:
        # same as above
        V_nospike += -t_step/tau*(V_nospike-E_L)+R*I_0
        P_nospike[bin_index] = P_nospike[bin_index]*(1-_P(V_nospike, V_T_dyn_nospike, DeltaV, lambda_0, t_step))
    return V_nospike


def _stochastic_spiking(V, t, past_spikes, bin_index, ref_count, counts, spiketimes, refractory_steps, t_step, settings):
    V_T = settings["V_T"]
    V_R = settings["V_R"]
    DeltaV = settings["DeltaV"]
    lambda_0 = settings["lambda_0"]
    T_ref = settings["T_ref"]
    # Count-down for refractory period
    if ref_count > 0:
        ref_count += -1
    else:
        V_T_dyn = V_T+_threshold_adaptation(past_spikes, settings)
        # print(_P(V, V_T_dyn, DeltaV, lambda_0, t_step))
        if random.random() < _P(V, V_T_dyn, DeltaV, lambda_0, t_step):
            # past spikes are stored with an offset T_ref, this offset is also used in Pozzorini et al 2013
            past_spikes = np.append(past_spikes, -T_ref)
            # reset membrane integrate_mebrane_potential
            V = V_R
            # initiate refractory period
            ref_count = refractory_steps
            # store spiketime and spike counts
            spiketimes += [t]
            counts[bin_index] += 1
    return V, ref_count, past_spikes


"""Model settings"""
settings = {}
# input current
settings["I_0"] = 0.00158
# membrane
settings["V_R"] = -38.8
settings["V_T"] = -51.9
settings["E_L"] = -69.4
settings["R"] = 485.
settings["tau"] = 0.0153
# spiking
settings["lambda_0"] = 2.0
settings["T_ref"] = 0.002
settings["DeltaV"] = 0.75
# threshold adaptation
settings["beta_thresh"] = 0.93
settings["a_thresh"] = 19.2
settings["T_thresh"] = 0.0083
tau = settings["tau"]
E_L = settings["E_L"]
R = settings["R"]
I_0 = settings["I_0"]
V_R = settings["V_R"]
V_T = settings["V_T"]


# simulation
t_step = 0.0005
t_bin = 0.005
number_steps_per_bin = int(t_bin/t_step)
refractory_steps = int(settings["T_ref"]/t_step)
# T=54100 # 10 times 90 min = 54000 s + 100 s burn in
T = 500.
N_steps = int(T/t_step)
N_bins = int(N_steps/number_steps_per_bin)

"""Simulation"""
# Data arrays
spiketimes = []
counts = np.zeros(N_bins)
P_nospike = np.zeros(N_bins)+1

# Initialisation
# past times are positive and give the time distance to the current simulation step
# Start with 80 random spikes in the last 22 seconds (roughly in agreement with 4 Hz firing)
past_spikes = np.sort([22.*random.random() for i in range(80)])[::-1]
len(past_spikes)
ref_count = 0
# start membrane potential at fixed point (dV/dt = 0)
V = R*I_0*tau/t_step+E_L
step_index = 0
t = 0
spike_count = 0
for step in np.arange(N_steps):
    # index of the current time bin
    bin_index = int(step_index/number_steps_per_bin)
    # do not store past spikes beyond 22 seconds
    if past_spikes[0] > 22:
        past_spikes = np.delete(past_spikes, 0, axis=0)
    # When completed a bin, then align the simulation with no spikes in the bin with the actual simulation
    if (step_index % number_steps_per_bin) == 0:
        V_nospike = V

    # Function integrate_mebrane_potential
    V = _integrate_membrane_potential(V, ref_count, settings, t_step)
    # Function compute_nospike_prob and integrates V_nospike
    V_nospike = _compute_nospike_prob(V_nospike,  P_nospike, counts, past_spikes, step_index, bin_index, ref_count,  number_steps_per_bin, t_step, settings)

    # Function stochastic_spiking
    V, ref_count, past_spikes = _stochastic_spiking(V, t, past_spikes, bin_index, ref_count, counts, spiketimes, refractory_steps, t_step, settings)

    # advance time
    past_spikes += t_step
    t += t_step
    step_index += 1

# Compute Rtot
P_spike = 1-P_nospike
H_spiking_cond = np.sum(-np.log(counts * P_spike + (1 - counts) * P_nospike)) / N_bins
P_spike_marginal = np.sum(counts) / float(N_bins)
H_spiking = -P_spike_marginal * np.log(P_spike_marginal) - (1 - P_spike_marginal) * np.log(1 - P_spike_marginal)
R_tot = 1 - H_spiking_cond / H_spiking


# TODO: Load and save again in numpy format
np.save('{}/analysis_data/spiketimes_simulation.npy'.format(codedirectory), spiketimes)
np.save('{}/analysis_data/R_tot_simulation.npy'.format(codedirectory), [R_tot])
