import os

from brian import pA, second, nS, Hz, ms
import h5py
import numpy as np
import matplotlib.pyplot as plt

from perceptchoice.model.monitor import SessionMonitor
from perceptchoice.model.network import default_params, pyr_params, simulation_params, AccumulatorNetwork
from perceptchoice.model.virtual_subject import VirtualSubject


def run_accumulator_virtual_subjects(subj_ids, conditions, output_dir, wta_params):
    # Run each subject
    for subj_id in subj_ids:
        print('***** Running subject %d *****' % subj_id)
        # network is more sensitive to background noise - need to use smaller range, wider operating range of firing rates
        # Resp threshold has to vary with background noise
        # maybe not choice hysteresis bc both pyramidal populations active - both bias the selection on next trial
        # also - since both are active, more excitatory drive on interneurons, therefore more inhibition of pyramidal
        # populations, rates decay faster
        wta_params.background_freq=855+(870-855)*np.random.random()
        mid_resp_threshold=wta_params.background_freq-835.0
        wta_params.resp_threshold=mid_resp_threshold+np.random.uniform(low=-1.0, high=1.0)

        # Set initial input weights and modify NMDA recurrent
        pyramidal_params=pyr_params(w_nmda=0.145*nS, w_ampa_ext_correct=1.6*nS, w_ampa_ext_incorrect=0.9*nS)

        # Create a virtual subject
        subject=VirtualSubject(subj_id, wta_params=wta_params, pyr_params=pyramidal_params, network_class=AccumulatorNetwork)

        # Run through each condition
        for condition, sim_params in conditions.iteritems():
            # Reinitialize state variables in subject network
            subject.net.reinit(states=True)
            # Run session
            output_file=os.path.join(output_dir, 'subject.%d.%s.h5' % (subj_id,condition))
            run_session(subject, condition, sim_params, output_file=output_file, continuous=True)


def run_virtual_subjects(subj_ids, conditions, output_dir, behavioral_param_file, wta_params,
                         continuous=True):
    """
    Runs a set of virtual subjects on the given conditions
    subj_ids = list of subject IDs
    conditions = dictionary: {condition name: simulation_params}
    output_dir = directory to store h5 output files
    behavioral_param_file = h5 file containing softmax-RL parameter distributions, background freq is sampled using
        inverse temp param distribution
    """
    # Load alpha and beta params of control group from behavioral parameter file
    f = h5py.File(behavioral_param_file)
    control_group=f['control']
    alpha_vals=np.array(control_group['alpha'])
    beta_vals=np.array(control_group['beta'])

    # Run each subject
    for subj_id in subj_ids:
        print('***** Running subject %d *****' % subj_id)

        # Sample beta from subject distribution - don't use subjects with high alpha
        beta_hist,beta_bins=np.histogram(beta_vals[np.where(alpha_vals<.99)[0]], density=True)
        bin_width=beta_bins[1]-beta_bins[0]
        beta_bin=np.random.choice(beta_bins[:-1], p=beta_hist*bin_width)
        beta=beta_bin+np.random.rand()*bin_width

        # Create virtual subject parameters - background freq from beta dist, resp threshold between 15 and 25Hz
        wta_params.background_freq=(beta-161.08)/-.17
        wta_params.resp_threshold=18+np.random.uniform(4)

        # Set initial input weights and modify NMDA recurrent
        pyramidal_params=pyr_params(w_nmda=0.145*nS, w_ampa_ext_correct=1.6*nS, w_ampa_ext_incorrect=0.9*nS)

        # Create a virtual subject
        subject=VirtualSubject(subj_id, wta_params=wta_params, pyr_params=pyramidal_params)

        # Run through each condition
        for condition, sim_params in conditions.iteritems():
            # Reinitialize state variables in subject network
            subject.net.reinit(states=True)
            # Run session
            output_file=os.path.join(output_dir, 'subject.%d.%s.h5' % (subj_id,condition))
            run_session(subject, condition, sim_params, output_file=output_file, continuous=continuous)


def run_session(subject, condition, sim_params, output_file=None, plot=False, continuous=True):
    """
    Run session in subject
    subject = subject object
    sim_params = simulation params
    output_file = if not none, writes h5 output to filename
    plot = plots session data if True
    """
    print('** Condition: %s **' % condition)

    # Create session monitor
    session_monitor=SessionMonitor(subject.wta_network, sim_params, conv_window=40,
        record_firing_rates=True)

    # Run on five coherence levels
    coherence_levels=[0.032, .064, .128, .256, .512]
    # Trials per coherence level
    trials_per_level=20
    # Create inputs for each trial
    trial_inputs=np.zeros((trials_per_level*len(coherence_levels),2))
    # Create left and right directions for each coherence level
    for i in range(len(coherence_levels)):
        coherence = coherence_levels[i]
        # Left
        min_idx=i*trials_per_level
        max_idx=i*trials_per_level+trials_per_level/2
        trial_inputs[min_idx:max_idx, 0] = subject.wta_params.mu_0 + subject.wta_params.p_a * coherence * 100.0
        trial_inputs[min_idx:max_idx, 1] = subject.wta_params.mu_0 - subject.wta_params.p_b * coherence * 100.0

        #Right
        min_idx=i*trials_per_level+trials_per_level/2
        max_idx=i*trials_per_level + trials_per_level
        trial_inputs[min_idx:max_idx, 0] = subject.wta_params.mu_0 - subject.wta_params.p_b * coherence * 100.0
        trial_inputs[min_idx:max_idx, 1] = subject.wta_params.mu_0 + subject.wta_params.p_a * coherence * 100.0

    # Shuffle trials
    trial_inputs=np.random.permutation(trial_inputs)

    # Simulate each trial
    for t in range(sim_params.ntrials):
        print('Trial %d' % t)

        # Get task input for trial and figure out which is correct
        task_input_rates=trial_inputs[t,:]
        correct_input=np.where(task_input_rates==np.max(task_input_rates))[0]

        # Run trial
        if not continuous:
            subject.net.reinit(states=True)
        subject.run_trial(sim_params, task_input_rates)

        #subject.wta_monitor.plot()
        #plt.show()

        # Record trial
        session_monitor.record_trial(t, task_input_rates, correct_input, subject.wta_network, subject.wta_monitor)

    # Write output
    if output_file is not None:
        session_monitor.write_output(output_file)

    # Plot
    if plot:
        if sim_params.ntrials>1:
            session_monitor.plot()
        else:
            subject.wta_monitor.plot()
        plt.show()


def run_main_conditions(data_path, behavioral_params_file):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    # Stimulation conditions
    conditions = {
        'control': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                     stim_start_time=1 * second,
                                     stim_end_time=2 * second),
        'depolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                          stim_start_time=1 * second,
                                          stim_end_time=2 * second, p_dcs=stim_intensity_max,
                                          i_dcs=-0.5 * stim_intensity_max,
                                          dcs_start_time=0 * second, dcs_end_time=3 * second),
        'hyperpolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                             stim_start_time=1 * second,
                                             stim_end_time=2 * second, p_dcs=-1 * stim_intensity_max,
                                             i_dcs=0.5 * stim_intensity_max,
                                             dcs_start_time=0 * second, dcs_end_time=3 * second)
    }
    wta_params=default_params()
    run_virtual_subjects(range(20), conditions, data_path, behavioral_params_file, wta_params)


def run_pyr_only_control_sim(data_path, behavioral_params_file):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    # Stimulation conditions
    conditions = {
        'control': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                     stim_start_time=1 * second,
                                     stim_end_time=2 * second),
        'depolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                          stim_start_time=1 * second,
                                          stim_end_time=2 * second, p_dcs=stim_intensity_max,
                                          i_dcs=0, dcs_start_time=0 * second, dcs_end_time=3 * second),
        'hyperpolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                             stim_start_time=1 * second,
                                             stim_end_time=2 * second, p_dcs=-1 * stim_intensity_max,
                                             i_dcs=0, dcs_start_time=0 * second, dcs_end_time=3 * second)
    }
    wta_params=default_params()
    run_virtual_subjects(range(20), conditions, data_path, behavioral_params_file, wta_params)


def run_inh_only_control_sim(data_path, behavioral_params_file):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    # Stimulation conditions
    conditions = {
        'control': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                     stim_start_time=1 * second,
                                     stim_end_time=2 * second),
        'depolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                          stim_start_time=1 * second,
                                          stim_end_time=2 * second, p_dcs=0, i_dcs=-0.5 * stim_intensity_max,
                                          dcs_start_time=0 * second, dcs_end_time=3 * second),
        'hyperpolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                             stim_start_time=1 * second,
                                             stim_end_time=2 * second, p_dcs=0, i_dcs=0.5 * stim_intensity_max,
                                             dcs_start_time=0 * second, dcs_end_time=3 * second)
    }
    wta_params=default_params()
    run_virtual_subjects(range(20), conditions, data_path, behavioral_params_file, wta_params)


def run_uniform_stim_control_sim(data_path, behavioral_params_file):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    # Stimulation conditions
    conditions = {
        'control': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                     stim_start_time=1 * second,
                                     stim_end_time=2 * second),
        'depolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                          stim_start_time=1 * second,
                                          stim_end_time=2 * second, p_dcs=stim_intensity_max,
                                          i_dcs=0.5 * stim_intensity_max,
                                          dcs_start_time=0 * second, dcs_end_time=3 * second),
        'hyperpolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                             stim_start_time=1 * second,
                                             stim_end_time=2 * second, p_dcs=-1 * stim_intensity_max,
                                             i_dcs=-0.5 * stim_intensity_max,
                                             dcs_start_time=0 * second, dcs_end_time=3 * second)
    }
    wta_params=default_params()
    run_virtual_subjects(range(20), conditions, data_path, behavioral_params_file, wta_params)


def run_mu_0_control_sim(mu_0, data_path, behavioral_params_file):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    # Stimulation conditions
    conditions = {
        'control': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                     stim_start_time=1 * second,
                                     stim_end_time=2 * second),
        'depolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                          stim_start_time=1 * second,
                                          stim_end_time=2 * second, p_dcs=stim_intensity_max,
                                          i_dcs=-0.5 * stim_intensity_max,
                                          dcs_start_time=0 * second, dcs_end_time=3 * second),
        'hyperpolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                             stim_start_time=1 * second,
                                             stim_end_time=2 * second, p_dcs=-1 * stim_intensity_max,
                                             i_dcs=0.5 * stim_intensity_max,
                                             dcs_start_time=0 * second, dcs_end_time=3 * second)
    }
    wta_params=default_params(mu_0=mu_0)
    run_virtual_subjects(range(20), conditions, data_path, behavioral_params_file, wta_params)


def run_refresh_rate_control_sim(refresh_rate, data_path, behavioral_params_file):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    # Stimulation conditions
    conditions = {
        'control': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                     stim_start_time=1 * second,
                                     stim_end_time=2 * second),
        'depolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                          stim_start_time=1 * second,
                                          stim_end_time=2 * second, p_dcs=stim_intensity_max,
                                          i_dcs=-0.5 * stim_intensity_max,
                                          dcs_start_time=0 * second, dcs_end_time=3 * second),
        'hyperpolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                             stim_start_time=1 * second,
                                             stim_end_time=2 * second, p_dcs=-1 * stim_intensity_max,
                                             i_dcs=0.5 * stim_intensity_max,
                                             dcs_start_time=0 * second, dcs_end_time=3 * second)
    }
    wta_params=default_params(refresh_rate=refresh_rate*Hz)
    run_virtual_subjects(range(20), conditions, data_path, behavioral_params_file, wta_params)


def run_reinit_control_sim(data_path, behavioral_params_file):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    # Stimulation conditions
    conditions = {
        'control': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                     stim_start_time=1 * second,
                                     stim_end_time=2 * second),
        'depolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                          stim_start_time=1 * second,
                                          stim_end_time=2 * second, p_dcs=stim_intensity_max,
                                          i_dcs=-0.5 * stim_intensity_max,
                                          dcs_start_time=0 * second, dcs_end_time=3 * second),
        'hyperpolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3 * second,
                                             stim_start_time=1 * second,
                                             stim_end_time=2 * second, p_dcs=-1 * stim_intensity_max,
                                             i_dcs=0.5 * stim_intensity_max,
                                             dcs_start_time=0 * second, dcs_end_time=3 * second)
    }
    wta_params=default_params()
    run_virtual_subjects(range(20), conditions, data_path, behavioral_params_file, wta_params, continuous=False)


def run_accumulator_control_sim(data_path):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    # Stimulation conditions
    conditions = {
        'control': simulation_params(
            ntrials=trials_per_condition,
            trial_duration=3 * second,
            stim_start_time=1 * second,
            stim_end_time=2 * second
        ),
        'depolarizing': simulation_params(
            ntrials=trials_per_condition,
            trial_duration=3 * second,
            stim_start_time=1 * second,
            stim_end_time=2 * second,
            p_dcs=stim_intensity_max,
            i_dcs=-0.5 * stim_intensity_max,
            dcs_start_time=0 * second,
            dcs_end_time=3 * second
        ),
        'hyperpolarizing': simulation_params(
            ntrials=trials_per_condition,
            trial_duration=3 * second,
            stim_start_time=1 * second,
            stim_end_time=2 * second,
            p_dcs=-1 * stim_intensity_max,
            i_dcs=0.5 * stim_intensity_max,
            dcs_start_time=0 * second,
            dcs_end_time=3 * second
        )
    }
    wta_params=default_params()
    run_accumulator_virtual_subjects(range(20), conditions, data_path, wta_params)


def run_isi_control_sim(isi, data_path, behavioral_params_file):
    # Trials per condition
    trials_per_condition = 100
    # Max stimulation intensity
    stim_intensity_max = 0.75 * pA
    start_time=(isi/2.0)*ms
    end_time=start_time+1*second
    duration=end_time+(isi/2.0)*ms
    # Stimulation conditions
    conditions = {
        'control': simulation_params(
            ntrials=trials_per_condition,
            trial_duration=duration,
            stim_start_time=start_time,
            stim_end_time=end_time
        ),
        'depolarizing': simulation_params(
            ntrials=trials_per_condition,
            trial_duration=duration,
            stim_start_time=start_time,
            stim_end_time=end_time,
            p_dcs=stim_intensity_max,
            i_dcs=-0.5 * stim_intensity_max,
            dcs_start_time=0 * second,
            dcs_end_time=duration
        ),
        'hyperpolarizing': simulation_params(
            ntrials=trials_per_condition,
            trial_duration=duration,
            stim_start_time=start_time,
            stim_end_time=end_time,
            p_dcs=-1 * stim_intensity_max,
            i_dcs=0.5 * stim_intensity_max,
            dcs_start_time=0 * second,
            dcs_end_time=duration
        )
    }
    wta_params=default_params()
    run_virtual_subjects(range(20), conditions, data_path, behavioral_params_file, wta_params)


if __name__=='__main__':
    run_main_conditions('/data/pySBI/rdmd/virtual_subjects2', '../../data/fitted_behavioral_params.h5')
