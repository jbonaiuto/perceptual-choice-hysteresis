import os
import math

from brian import second, Parameters, farad, siemens, volt, Hz, amp, ms
import h5py
from matplotlib.mlab import normpdf
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import wilcoxon, norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from perceptchoice.utils import FitSigmoid, mdm_outliers, FitWeibull, FitRT


colors={
    'control': 'b',
    'depolarizing': 'r',
    'hyperpolarizing': 'g'
}

coherences=[0.0320, 0.0640, 0.1280, 0.2560, 0.5120]

conditions=['control','depolarizing','hyperpolarizing']

def read_subject_data(data_dir, subj_id, conditions):
    subj_data={}
    for condition in conditions:
        f = h5py.File(os.path.join(data_dir,'subject.%d.%s.h5' % (subj_id,condition)),'r')

        f_network_params = f['network_params']
        network_params=Parameters(
            # Neuron parameters
            C = float(f_network_params.attrs['C']) * farad,
            gL = float(f_network_params.attrs['gL']) * siemens,
            EL = float(f_network_params.attrs['EL']) * volt,
            VT = float(f_network_params.attrs['VT']) * volt,
            DeltaT = float(f_network_params.attrs['DeltaT']) * volt,
            Vr = float(f_network_params.attrs['Vr']) * volt,
            Mg = float(f_network_params.attrs['Mg']),
            E_ampa = float(f_network_params.attrs['E_ampa'])*volt,
            E_nmda = float(f_network_params.attrs['E_nmda'])*volt,
            E_gaba_a = float(f_network_params.attrs['E_gaba_a'])*volt,
            tau_ampa = float(f_network_params.attrs['tau_ampa'])*second,
            tau1_nmda = float(f_network_params.attrs['tau1_nmda'])*second,
            tau2_nmda = float(f_network_params.attrs['tau2_nmda'])*second,
            tau_gaba_a = float(f_network_params.attrs['tau_gaba_a'])*second,
            p_e_e=float(f_network_params.attrs['p_e_e']),
            p_e_i=float(f_network_params.attrs['p_e_i']),
            p_i_i=float(f_network_params.attrs['p_i_i']),
            p_i_e=float(f_network_params.attrs['p_i_e']),
            background_freq=float(f_network_params.attrs['background_freq'])*Hz,
            input_var=float(f_network_params.attrs['input_var'])*Hz,
            refresh_rate=float(f_network_params.attrs['refresh_rate'])*Hz,
            num_groups=int(f_network_params.attrs['num_groups']),
            network_group_size=int(f_network_params.attrs['network_group_size']),
            background_input_size=int(f_network_params.attrs['background_input_size']),
            mu_0=float(f_network_params.attrs['mu_0']),
            f=float(f_network_params.attrs['f']),
            task_input_resting_rate=float(f_network_params.attrs['task_input_resting_rate'])*Hz,
            resp_threshold=float(f_network_params.attrs['resp_threshold']),
            p_a=float(f_network_params.attrs['p_a']),
            p_b=float(f_network_params.attrs['p_b']),
            task_input_size=int(f_network_params.attrs['task_input_size'])
        )

        f_sim_params = f['sim_params']
        sim_params=Parameters(
            trial_duration=float(f_sim_params.attrs['trial_duration'])*second,
            stim_start_time=float(f_sim_params.attrs['stim_start_time'])*second,
            stim_end_time=float(f_sim_params.attrs['stim_end_time'])*second,
            dt=float(f_sim_params.attrs['dt'])*second,
            ntrials=int(f_sim_params.attrs['ntrials']),
            p_dcs=float(f_sim_params.attrs['p_dcs'])*amp,
            i_dcs=float(f_sim_params.attrs['i_dcs'])*amp,
            dcs_start_time=float(f_sim_params.attrs['dcs_start_time'])*second,
            dcs_end_time=float(f_sim_params.attrs['dcs_end_time'])*second,
        )

        f_behav = f['behavior']
        trial_rt = np.array(f_behav['trial_rt'])
        trial_resp = np.array(f_behav['trial_resp'])
        trial_correct = np.array(f_behav['trial_correct'])

        f_neur = f['neural']
        trial_inputs = np.array(f_neur['trial_inputs'])
        trial_data = []
        trial_rates={
            'inhibitory_rate':[],
            'excitatory_rate_0':[],
            'excitatory_rate_1':[],
        }

        f_rates=f_neur['firing_rates']
        for trial_idx in range(trial_rt.shape[1]):
            f_trial=f_rates['trial_%d' % trial_idx]
            trial_rates['inhibitory_rate'].append(np.array(f_trial['inhibitory_rate']))
            trial_rates['excitatory_rate_0'].append(np.array(f_trial['excitatory_rate_0']))
            trial_rates['excitatory_rate_1'].append(np.array(f_trial['excitatory_rate_1']))

        last_resp = float('NaN')
        last_coherence = float('NaN')
        last_correct=float('NaN')
        last_direction=float('NaN')
        for trial_idx in range(trial_rt.shape[1]):
            direction = np.where(trial_inputs[:, trial_idx] == np.max(trial_inputs[:, trial_idx]))[0][0]
            if direction == 0:
                direction = -1
            coherence = np.abs((trial_inputs[0, trial_idx] - network_params.mu_0) / (network_params.p_a * 100.0))
            coherence_diffs=np.abs(np.array(coherences)-coherence)
            coherence=coherences[np.where(coherence_diffs==np.min(coherence_diffs))[0][0]]
            correct = int(trial_correct[0, trial_idx])
            resp = int(trial_resp[0, trial_idx])
            if resp == -1:
                resp=float('NaN')
            elif resp == 0:
                resp = -1
            rt = trial_rt[0, trial_idx]

            trial_data.append([trial_idx, direction, coherence, correct, resp, last_resp, rt, last_coherence, last_correct, last_direction])
            last_resp = resp
            last_coherence=coherence
            last_correct=correct
            last_direction=direction
        trial_data = np.array(trial_data)
        outliers=mdm_outliers(trial_data[:,6])
        trial_data[outliers,4:5]=float('NaN')
        subj_data[condition]={
            'behavior': trial_data,
            'neural': trial_rates,
            'sim_params': sim_params,
            'network_params': network_params
        }

    return subj_data


def analyze_bias(subjects, output_dir):
    condition_repeat_choice_biases={}
    condition_different_choice_biases={}
    for subject in subjects:
        subj_repeat_choice_biases, subj_different_choice_biases=analyze_subject_bias(subject)
        for condition in conditions:
            if not condition in condition_repeat_choice_biases:
                condition_repeat_choice_biases[condition]=[]
                condition_different_choice_biases[condition]=[]
            condition_repeat_choice_biases[condition].append(np.mean(np.array(subj_repeat_choice_biases[condition])))
            condition_different_choice_biases[condition].append(np.mean(np.array(subj_different_choice_biases[condition])))

    fig=plt.figure()
    ax=fig.add_subplot(2,1,1)
    cond_means=[]
    cond_stderr=[]
    for condition in conditions:
        cond_means.append(np.mean(condition_repeat_choice_biases[condition]))
        cond_stderr.append(np.std(condition_repeat_choice_biases[condition])/np.sqrt(len(condition_repeat_choice_biases[condition])))
    ind=np.arange(len(conditions))
    width=0.75
    rects=ax.bar(ind, cond_means, width, yerr=cond_stderr, error_kw=dict(ecolor='black'))
    for idx,rect in enumerate(rects):
        rect.set_color(colors[conditions[idx]])
    ax.set_ylim([0,2.5])
    ax.set_xlim([np.min(ind)-.1,np.max(ind)+width+.1])
    ax.set_ylabel('Bias')
    ax.set_xticks(ind+.5*width)
    ax.set_xticklabels(conditions)

    ax=fig.add_subplot(2,1,2)
    cond_means=[]
    cond_stderr=[]
    for condition in conditions:
        cond_means.append(np.abs(np.mean(condition_different_choice_biases[condition])))
        cond_stderr.append(np.std(condition_different_choice_biases[condition])/np.sqrt(len(condition_different_choice_biases[condition])))
    ind=np.arange(len(conditions))
    width=0.75
    rects=ax.bar(ind, cond_means, width, yerr=cond_stderr, error_kw=dict(ecolor='black'))
    for idx,rect in enumerate(rects):
        rect.set_color(colors[conditions[idx]])
    ax.set_xlim([np.min(ind)-.1,np.max(ind)+width+.1])
    ax.set_ylim([0,2.5])
    ax.set_ylabel('Bias')
    ax.set_xticks(ind+.5*width)
    ax.set_xticklabels(conditions)

    print('Repeat choice')
    for condition in ['depolarizing','hyperpolarizing']:
        (W,p)=wilcoxon(condition_repeat_choice_biases['control'], condition_repeat_choice_biases[condition])
        N=len(subjects)
        print('%s, W(%d)=%.4f, p=%.6f' % (condition, N-1,W,p))

    print('Different choice')
    for condition in ['depolarizing','hyperpolarizing']:
        (W,p)=wilcoxon(condition_different_choice_biases['control'], condition_different_choice_biases[condition])
        N=len(subjects)
        print('%s, W(%d)=%.4f, p=%.6f' % (condition, N-1,W,p))

    out_file=file(os.path.join(output_dir,'bias_repeated_choice.csv'),'w')
    out_file.write('VirtualSubjID')
    for condition in condition_repeat_choice_biases:
        out_file.write(',%s' % condition)
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for condition in condition_repeat_choice_biases:
            out_file.write(',%.4f' % condition_repeat_choice_biases[condition][subj_idx])
        out_file.write('\n')
    out_file.close()

    out_file=file(os.path.join(output_dir,'bias_different_choice.csv'),'w')
    out_file.write('VirtualSubjID')
    for condition in condition_different_choice_biases:
        out_file.write(',%s' % condition)
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for condition in condition_different_choice_biases:
            out_file.write(',%.4f' % condition_different_choice_biases[condition][subj_idx])
        out_file.write('\n')
    out_file.close()


def analyze_subject_bias(subject):
    repeat_choice_biases={}
    different_choice_biases={}
    for condition,subj_data in subject.iteritems():
        repeat_choice_biases[condition]=[]
        different_choice_biases[condition]=[]
        trial_data=subj_data['behavior']
        trial_rates=subj_data['neural']
        time_ticks=(np.array(range(trial_rates['excitatory_rate_0'][0].shape[0]))*subj_data['sim_params'].dt)/ms
        bias_time=np.intersect1d(np.where(time_ticks>=500)[0],np.where(time_ticks<1000)[0])
        # For each trial
        for trial_idx in range(1,trial_data.shape[0]):
            last_resp=trial_data[trial_idx,5]
            resp=trial_data[trial_idx,4]

            if not math.isnan(resp) and not math.isnan(last_resp):
                if resp==-1:
                    bias=trial_rates['excitatory_rate_0'][trial_idx][bias_time]-trial_rates['excitatory_rate_1'][trial_idx][bias_time]
                else:
                    bias=trial_rates['excitatory_rate_1'][trial_idx][bias_time]-trial_rates['excitatory_rate_0'][trial_idx][bias_time]
                if resp==last_resp:
                    repeat_choice_biases[condition].append(bias)
                else:
                    different_choice_biases[condition].append(bias)
    return repeat_choice_biases, different_choice_biases


def analyze_subject_accuracy_rt(subject):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    for condition,subj_data in subject.iteritems():
        trial_data=subj_data['behavior']
        condition_coherence_accuracy[condition]={}
        condition_coherence_rt[condition]={}
        # For each trial
        for trial_idx in range(trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]
            correct=trial_data[trial_idx,3]
            rt=trial_data[trial_idx,6]

            if not math.isnan(rt):
                if not coherence in condition_coherence_accuracy[condition]:
                    condition_coherence_accuracy[condition][coherence]=[]
                condition_coherence_accuracy[condition][np.abs(coherence)].append(float(correct))

                if not coherence in condition_coherence_rt[condition]:
                    condition_coherence_rt[condition][coherence]=[]
                condition_coherence_rt[condition][np.abs(coherence)].append(rt)

        coherences = sorted(condition_coherence_accuracy[condition].keys())
        accuracy=[]
        for coherence in coherences:
            accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
        acc_fit = FitWeibull(coherences, accuracy, guess=[0.0, 0.2], display=0)
        condition_accuracy_thresh[condition]=acc_fit.inverse(0.8)


    for stim_condition in ['depolarizing', 'hyperpolarizing']:
        condition_coherence_rt_diff[stim_condition]={}
        coherences=sorted(condition_coherence_rt[stim_condition].keys())
        for coherence in coherences:
            condition_coherence_rt_diff[stim_condition][coherence]=np.mean(condition_coherence_rt[stim_condition][coherence])-np.mean(condition_coherence_rt['control'][coherence])

    return condition_coherence_accuracy, condition_coherence_rt, condition_coherence_rt_diff, condition_accuracy_thresh


def analyze_subject_choice_hysteresis(subject):
    condition_coherence_choices={
        'L*': {},
        'R*': {}
    }
    condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    condition_logistic_params={
        'a0': {},
        'a1': {},
        'a2': {}
    }
    for condition,subj_data in subject.iteritems():
        trial_data=subj_data['behavior']
        # Dict of coherence levels
        condition_coherence_choices['L*'][condition]={}
        condition_coherence_choices['R*'][condition]={}

        # For each trial
        for trial_idx in range(1,trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]*trial_data[trial_idx,1]
            last_resp=trial_data[trial_idx,5]
            resp=trial_data[trial_idx,4]

            if not math.isnan(resp) and not math.isnan(last_resp):

                if last_resp<0:
                    if not coherence in condition_coherence_choices['L*'][condition]:
                        condition_coherence_choices['L*'][condition][coherence]=[]
                        # Append 0 to list if left (-1) or 1 if right
                    condition_coherence_choices['L*'][condition][coherence].append(np.max([0,resp]))
                elif last_resp>0:
                    # List of rightward choices (0=left, 1=right)
                    if not coherence in condition_coherence_choices['R*'][condition]:
                        condition_coherence_choices['R*'][condition][coherence]=[]
                        # Append 0 to list if left (-1) or 1 if right
                    condition_coherence_choices['R*'][condition][coherence].append(np.max([0,resp]))

        choice_probs=[]
        full_coherences=[]
        for coherence in condition_coherence_choices['L*'][condition]:
            choice_probs.append(np.mean(condition_coherence_choices['L*'][condition][coherence]))
            full_coherences.append(coherence)
        acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
        condition_sigmoid_offsets['L*'][condition]=acc_fit.inverse(0.5)

        choice_probs=[]
        full_coherences=[]
        for coherence in condition_coherence_choices['R*'][condition]:
            choice_probs.append(np.mean(condition_coherence_choices['R*'][condition][coherence]))
            full_coherences.append(coherence)
        acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
        condition_sigmoid_offsets['R*'][condition]=acc_fit.inverse(0.5)

        data=pd.DataFrame({
            'resp': np.clip(trial_data[1:,4],0,1),
            'coh': trial_data[1:,2]*trial_data[1:,1],
            'last_resp': trial_data[1:,5]
        })
        data['intercept']=1.0

        data=data.dropna(axis=0)
        logit = sm.Logit(data['resp'], data[['coh','last_resp','intercept']])
        result = logit.fit(method='bfgs',disp=False)
        condition_logistic_params['a0'][condition]=result.params['intercept']
        condition_logistic_params['a1'][condition]=result.params['coh']
        condition_logistic_params['a2'][condition]=result.params['last_resp']

    return condition_coherence_choices, condition_sigmoid_offsets, condition_logistic_params


def analyze_accuracy_rt(subjects, output_dir):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    # For each subject
    for subject in subjects:

        subj_condition_coherence_accuracy, subj_condition_coherence_rt, subj_condition_coherence_rt_diff,\
            subj_condition_accuracy_thresh=analyze_subject_accuracy_rt(subject)

        for condition in conditions:
            if not condition in condition_coherence_accuracy:
                condition_coherence_accuracy[condition]={}
                condition_coherence_rt[condition]={}
                condition_accuracy_thresh[condition]=[]
            condition_accuracy_thresh[condition].append(subj_condition_accuracy_thresh[condition])

            for coherence in subj_condition_coherence_accuracy[condition]:
                if not coherence in condition_coherence_accuracy[condition]:
                    condition_coherence_accuracy[condition][coherence]=[]
                condition_coherence_accuracy[condition][coherence].append(np.mean(subj_condition_coherence_accuracy[condition][coherence]))

            for coherence in subj_condition_coherence_rt[condition]:
                if not coherence in condition_coherence_rt[condition]:
                    condition_coherence_rt[condition][coherence]=[]
                condition_coherence_rt[condition][coherence].append(np.mean(subj_condition_coherence_rt[condition][coherence]))

        for condition in subj_condition_coherence_rt_diff:
            if not condition in condition_coherence_rt_diff:
                condition_coherence_rt_diff[condition]={}
            for coherence in subj_condition_coherence_rt_diff[condition]:
                if not coherence in condition_coherence_rt_diff[condition]:
                    condition_coherence_rt_diff[condition][coherence]=[]
                condition_coherence_rt_diff[condition][coherence].append(subj_condition_coherence_rt_diff[condition][coherence])

    thresh_std={}
    for condition in conditions:
        thresh_std[condition]=np.std(condition_accuracy_thresh[condition])/np.sqrt(len(subjects))
    plot_choice_accuracy(colors, condition_coherence_accuracy, plot_err=True, thresh_std=thresh_std)

    plot_choice_rt_scaled(colors, condition_coherence_rt)

    plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=True)


    out_file=file(os.path.join(output_dir,'accuracy.csv'),'w')
    out_file.write('VirtualSubjID')
    for condition in condition_coherence_accuracy:
        for coherence in sorted(condition_coherence_accuracy[condition]):
            out_file.write(',%sCoherence%.4fAccuracy' % (condition,coherence))
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for condition in condition_coherence_accuracy:
            for coherence in sorted(condition_coherence_accuracy[condition]):
                out_file.write(',%.4f' % condition_coherence_accuracy[condition][coherence][subj_idx])
        out_file.write('\n')
    out_file.close()

    out_file=file(os.path.join(output_dir,'rt.csv'),'w')
    out_file.write('VirtualSubjID')
    for condition in condition_coherence_rt:
        for coherence in sorted(condition_coherence_rt[condition]):
            out_file.write(',%sCoherence%.4fRT' % (condition,coherence))
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for condition in condition_coherence_rt:
            for coherence in sorted(condition_coherence_rt[condition]):
                out_file.write(',%.4f' % condition_coherence_rt[condition][coherence][subj_idx])
        out_file.write('\n')
    out_file.close()

    out_file=file(os.path.join(output_dir,'rt_diff.csv'),'w')
    out_file.write('VirtualSubjID')
    for condition in condition_coherence_rt_diff:
        for coherence in sorted(condition_coherence_rt_diff[condition]):
            out_file.write(',%sCoherence%.4fRTDiff' % (condition,coherence))
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for condition in condition_coherence_rt_diff:
            for coherence in sorted(condition_coherence_rt_diff[condition]):
                out_file.write(',%.4f' % condition_coherence_rt_diff[condition][coherence][subj_idx])
        out_file.write('\n')
    out_file.close()

    thresh_results={
        'depolarizing': {},
        'hyperpolarizing': {},
    }
    (thresh_results['hyperpolarizing']['W'],thresh_results['hyperpolarizing']['p'])=wilcoxon(condition_accuracy_thresh['control'],condition_accuracy_thresh['hyperpolarizing'])
    (thresh_results['depolarizing']['W'],thresh_results['depolarizing']['p'])=wilcoxon(condition_accuracy_thresh['control'],condition_accuracy_thresh['depolarizing'])

    rtdiff_results={
        'depolarizing': {'coh': {}, 'intercept':{}},
        'hyperpolarizing': {'coh': {}, 'intercept':{}}
    }
    for condition in ['depolarizing','hyperpolarizing']:
        coh=[]
        rt_diff=[]
        for coherence in condition_coherence_rt_diff[condition]:
            for diff in condition_coherence_rt_diff[condition][coherence]:
                coh.append(coherence)
                rt_diff.append(diff)
        data=pd.DataFrame({
            'coh': coh,
            'rt_diff': rt_diff
        })
        data['intercept']=1.0
        lr = sm.GLM(data['rt_diff'], data[['coh','intercept']])
        result = lr.fit()
        for param in ['coh','intercept']:
            rtdiff_results[condition.lower()][param]['x']=result.params[param]
            rtdiff_results[condition.lower()][param]['t']=result.tvalues[param]
            rtdiff_results[condition.lower()][param]['p']=result.pvalues[param]

    print('Accuracy Threshold')
    for condition, results in thresh_results.iteritems():
        N=len(subjects)
        print('%s: W(%d)=%.4f, p=%.4f' % (condition, N-1, results['W'], results['p']))

    print('')
    print('RT Diff')
    for condition, results in rtdiff_results.iteritems():
        print('%s, coherence: B1=%.4f, t=%.4f, p=%.4f' % (condition, results['coh']['x'], results['coh']['t'],results['coh']['p']))
    print('')




def analyze_choice_hysteresis(subjects, output_dir):
    condition_coherence_choices={
        'L*': {},
        'R*': {}
    }
    condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    condition_logistic_params={
        'a0': {},
        'a1': {},
        'a2': {}
    }

    # For each subject
    for idx,subject in enumerate(subjects):
        subj_condition_coherence_choices, subj_condition_sigmoid_offsets, subj_condition_logistic_params=analyze_subject_choice_hysteresis(subject)

        for condition in conditions:
            if not condition in condition_coherence_choices['L*']:
                condition_coherence_choices['L*'][condition]={}
                condition_coherence_choices['R*'][condition]={}
                condition_sigmoid_offsets['L*'][condition]=[]
                condition_sigmoid_offsets['R*'][condition]=[]
                condition_logistic_params['a0'][condition]=[]
                condition_logistic_params['a1'][condition]=[]
                condition_logistic_params['a2'][condition]=[]

            condition_sigmoid_offsets['L*'][condition].append(subj_condition_sigmoid_offsets['L*'][condition])
            condition_sigmoid_offsets['R*'][condition].append(subj_condition_sigmoid_offsets['R*'][condition])

            condition_logistic_params['a0'][condition].append(subj_condition_logistic_params['a0'][condition])
            condition_logistic_params['a1'][condition].append(subj_condition_logistic_params['a1'][condition])
            condition_logistic_params['a2'][condition].append(subj_condition_logistic_params['a2'][condition])

            for coherence in subj_condition_coherence_choices['L*'][condition]:
                if not coherence in condition_coherence_choices['L*'][condition]:
                    condition_coherence_choices['L*'][condition][coherence]=[]
                condition_coherence_choices['L*'][condition][coherence].append(np.mean(subj_condition_coherence_choices['L*'][condition][coherence]))

            for coherence in subj_condition_coherence_choices['R*'][condition]:
                if not coherence in condition_coherence_choices['R*'][condition]:
                    condition_coherence_choices['R*'][condition][coherence]=[]
                condition_coherence_choices['R*'][condition][coherence].append(np.mean(subj_condition_coherence_choices['R*'][condition][coherence]))

    plot_indifference_hist(colors, condition_sigmoid_offsets)

    plot_choice_probability(colors, condition_coherence_choices)

    plot_logistic_parameter_ratio(colors, condition_logistic_params)

    out_file=file(os.path.join(output_dir,'indifference.csv'),'w')
    out_file.write('VirtualSubjID')
    for direction in ['L*','R*']:
        for condition in condition_sigmoid_offsets[direction]:
            out_file.write(',%s%s' % (direction,condition))
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for direction in ['L*','R*']:
            for condition in condition_sigmoid_offsets[direction]:
                out_file.write(',%.4f' % condition_sigmoid_offsets[direction][condition][subj_idx])
        out_file.write('\n')
    out_file.close()

    out_file=file(os.path.join(output_dir,'control_prob_choose_right.csv'),'w')
    out_file.write('VirtualSubjID')
    for direction in ['L*','R*']:
        for coherence in sorted(condition_coherence_choices[direction]['control']):
            out_file.write(',%sCoherence%.4f' % (direction,coherence))
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for direction in ['L*','R*']:
            for coherence in sorted(condition_coherence_choices[direction]['control']):
                prob=0.0
                if subj_idx<len(condition_coherence_choices[direction]['control'][coherence]):
                    prob=condition_coherence_choices[direction]['control'][coherence][subj_idx]
                out_file.write(',%.4f' % prob)
        out_file.write('\n')
    out_file.close()

    out_file=file(os.path.join(output_dir,'logistic.csv'),'w')
    out_file.write('VirtualSubjID')
    for param in ['a1','a2']:
        for condition in condition_logistic_params[param]:
            out_file.write(',%s%s' % (param,condition))
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for param in ['a1','a2']:
            for condition in condition_logistic_params[param]:
                out_file.write(',%.4f' % condition_logistic_params[param][condition][subj_idx])
        out_file.write('\n')
    out_file.close()

    indec_results={
        'depolarizing': {},
        'hyperpolarizing': {},
        'control': {}
    }
    (indec_results['control']['W'],indec_results['control']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['control'])-np.array(condition_sigmoid_offsets['R*']['control']))
    (indec_results['hyperpolarizing']['W'],indec_results['hyperpolarizing']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['control'])-np.array(condition_sigmoid_offsets['R*']['control']),
        np.array(condition_sigmoid_offsets['L*']['hyperpolarizing'])-np.array(condition_sigmoid_offsets['R*']['hyperpolarizing']))
    (indec_results['depolarizing']['W'],indec_results['depolarizing']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['control'])-np.array(condition_sigmoid_offsets['R*']['control']),
        np.array(condition_sigmoid_offsets['L*']['depolarizing'])-np.array(condition_sigmoid_offsets['R*']['depolarizing']))

    log_results={
        'depolarizing': {},
        'hyperpolarizing': {},
        'control': {},
    }
    control_ratio=np.array(condition_logistic_params['a2']['control'])/np.array(condition_logistic_params['a1']['control'])
    (log_results['control']['W'],log_results['control']['p'])=wilcoxon(control_ratio)

    anode_ratio=np.array(condition_logistic_params['a2']['depolarizing'])/np.array(condition_logistic_params['a1']['depolarizing'])
    (log_results['depolarizing']['W'],log_results['depolarizing']['p'])=wilcoxon(control_ratio, anode_ratio)

    cathode_ratio=np.array(condition_logistic_params['a2']['hyperpolarizing'])/np.array(condition_logistic_params['a1']['hyperpolarizing'])
    (log_results['hyperpolarizing']['W'],log_results['hyperpolarizing']['p'])=wilcoxon(control_ratio, cathode_ratio)

    print('Indecision Points')
    for condition, results in indec_results.iteritems():
        N=len(subjects)
        print('%s, W(%d)=%.4f, p=%.6f' % (condition,N-1,results['W'],results['p']))

    print('')
    print('Logistic Regression')
    for condition, results in log_results.iteritems():
        N=len(subjects)
        print('%s, W(%d)=%.4f, p=%.4f' % (condition, N-1, results['W'],results['p']))
    print('')


def plot_indifference_hist(colors, condition_sigmoid_offsets):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    binwidth=0.03
    lims=[-.2,.4]
    xx=np.arange(lims[0],lims[1],0.001)
    for condition in conditions:
        diff=np.array(condition_sigmoid_offsets['L*'][condition])-np.array(condition_sigmoid_offsets['R*'][condition])
        bins=np.arange(min(diff), max(diff)+binwidth, binwidth)
        hist,edges=np.histogram(diff, bins=bins)
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist/float(len(diff))*100.0, color=colors[condition], alpha=0.75, label=condition, width=binwidth)
        (mu, sigma) = norm.fit(diff)
        y = normpdf(xx, mu, sigma)*binwidth*100.0
        ax.plot(xx, y,'%s--' % colors[condition], linewidth=2)
    ax.set_xlim(lims)
    ax.set_ylim([0,30])
    ax.set_xlabel('Left*-Right* Indifference')
    ax.set_ylabel('% subjects')
    ax.legend(loc='best')


def plot_choice_accuracy(colors, condition_coherence_accuracy, plot_err=False, thresh_std=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for condition in conditions:
        coherences = sorted(condition_coherence_accuracy[condition].keys())
        mean_accuracy=[]
        stderr_accuracy=[]
        for coherence in coherences:
            mean_accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
            if plot_err:
                stderr_accuracy.append(np.std(condition_coherence_accuracy[condition][coherence])/np.sqrt(len(condition_coherence_accuracy[condition][coherence])))
        acc_fit = FitWeibull(coherences, mean_accuracy, guess=[0.0, 0.2], display=0)
        smoothInt = np.arange(.01, 1.0, 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        ax.semilogx(smoothInt, smoothResp, colors[condition], label=condition)
        if plot_err:
            ax.errorbar(coherences, mean_accuracy, yerr=stderr_accuracy, fmt='o%s' % colors[condition])
        else:
            ax.plot(coherences, mean_accuracy, 'o%s' % colors[condition])
        thresh=acc_fit.inverse(0.8)
        ax.plot([thresh,thresh],[0.5,1],'--%s' % colors[condition])
        if thresh_std is not None:
            rect=Rectangle((thresh-.5*thresh_std[condition],0.5),thresh_std[condition], 0.5, alpha=0.25, facecolor=colors[condition], edgecolor='none')
            ax.add_patch(rect)
    ax.legend(loc='best')
    ax.set_xlim([0.01,1.0])
    ax.set_ylim([0.5,1.0])
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% Correct')


def plot_choice_rt_scaled(colors, condition_coherence_rt):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    control_coherences=sorted(condition_coherence_rt['control'].keys())
    control_mean_rt=[np.mean(condition_coherence_rt['control'][x]) for x in control_coherences]
    scale=1/(np.max(control_mean_rt)-np.min(control_mean_rt))
    for condition in conditions:
        coherences = sorted(condition_coherence_rt[condition].keys())
        mean_rt=[]
        stderr_rt=[]
        for coherence in coherences:
            mean_rt.append(scale*(np.mean(condition_coherence_rt[condition][coherence])-np.min(control_mean_rt)))
            stderr_rt.append(scale*np.std(condition_coherence_rt[condition][coherence])/np.sqrt(len(condition_coherence_rt[condition][coherence])))
        rt_fit = FitRT(coherences, mean_rt, guess=[1,1,1], display=0)
        smoothInt = np.arange(.01, 1.0, 0.001)
        smoothRT = rt_fit.eval(smoothInt)
        ax.semilogx(smoothInt, smoothRT, colors[condition], label=condition)
        ax.errorbar(coherences, mean_rt, yerr=stderr_rt,fmt='o%s' % colors[condition])
    ax.set_xlim([0.01,1.0])
    ax.set_ylim([-0.2, 1.6])
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT')


def plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=False):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for stim_condition in ['depolarizing', 'hyperpolarizing']:
        coherences = np.array(sorted(condition_coherence_rt_diff[stim_condition].keys()))
        mean_diff=[]
        stderr_diff=[]
        for coherence in coherences:
            mean_diff.append(np.mean(condition_coherence_rt_diff[stim_condition][coherence]))
            if plot_err:
                stderr_diff.append(np.std(condition_coherence_rt_diff[stim_condition][coherence])/np.sqrt(len(condition_coherence_rt_diff[stim_condition][coherence])))
        mean_diff=np.array(mean_diff)

        clf = LinearRegression()
        clf.fit(np.expand_dims(coherences,axis=1),np.expand_dims(mean_diff,axis=1))
        a = clf.coef_[0][0]
        b = clf.intercept_[0]
        ax.plot([np.min(coherences), np.max(coherences)], [a * np.min(coherences) + b, a * np.max(coherences) + b], '--%s' % colors[stim_condition])

        if plot_err:
            ax.errorbar(coherences, mean_diff, yerr=stderr_diff, fmt='o%s' % colors[stim_condition], label=stim_condition)
        else:
            ax.plot(coherences, mean_diff, 'o%s' % colors[stim_condition], label=stim_condition)
    ax.legend(loc='best')
    ax.set_xlim([0,0.55])
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT Difference')


def plot_choice_probability(colors, condition_coherence_choices):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    condition='control'
    left_coherences=sorted(condition_coherence_choices['L*'][condition].keys())
    right_coherences=sorted(condition_coherence_choices['R*'][condition].keys())
    left_choice_probs = []
    right_choice_probs = []
    for coherence in left_coherences:
        left_choice_probs.append(np.mean(condition_coherence_choices['L*'][condition][coherence]))
    for coherence in right_coherences:
        right_choice_probs.append(np.mean(condition_coherence_choices['R*'][condition][coherence]))
    acc_fit = FitSigmoid(left_coherences, left_choice_probs, guess=[0.0, 0.2], display=0)
    smoothInt = np.arange(min(left_coherences), max(left_coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, '--%s' % colors[condition], label='Left*')
    ax.plot(left_coherences, left_choice_probs, 'o%s' % colors[condition])
    acc_fit = FitSigmoid(right_coherences, right_choice_probs, guess=[0.0, 0.2], display=0)
    smoothInt = np.arange(min(right_coherences), max(right_coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, colors[condition], label='Right*')
    ax.plot(right_coherences, right_choice_probs, 'o%s' % colors[condition])
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% of Right Choices')


def plot_logistic_parameter_ratio(colors, condition_logistic_params):
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xx=np.arange(-.512,.512,.001)
    condition='control'
    mean_a0=np.mean(condition_logistic_params['a0'][condition])
    mean_a1=np.mean(condition_logistic_params['a1'][condition])
    mean_a2=np.mean(condition_logistic_params['a2'][condition])
    yy_r=1/(1+np.exp(-(mean_a0+mean_a1*xx+mean_a2*1)))
    yy_l=1/(1+np.exp(-(mean_a0+mean_a1*xx+mean_a2*-1)))
    ax.plot(xx,yy_l,'%s--' % colors[condition], linewidth=2, label='Left*')
    ax.plot(xx,yy_r,colors[condition], linewidth=2, label='Right*')
    ax.legend(loc='best')
    ax.set_xlabel('coherence')
    ax.set_ylabel('P(R)')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lims=[-.1,.2]
    xx=np.arange(lims[0],lims[1],0.001)
    binwidth=.02
    for condition in conditions:
        ratio=np.array(condition_logistic_params['a2'][condition]) / np.array(condition_logistic_params['a1'][condition])
        bins=np.arange(min(ratio), max(ratio) + binwidth, binwidth)
        hist,edges=np.histogram(ratio, bins=bins)
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist/float(len(ratio))*100.0, color=colors[condition], alpha=0.75, label=condition, width=binwidth)
        (mu, sigma) = norm.fit(ratio)
        y = normpdf(xx, mu, sigma)*binwidth*100.0
        ax.plot(xx, y, '%s--' % colors[condition], linewidth=2)
    ax.legend(loc='best')
    ax.set_xlim(lims)
    ax.set_ylim([0, 35])
    ax.set_xlabel('a2/a1')
    ax.set_ylabel('% subjects')


if __name__=='__main__':
    #data_dir='/data/pySBI/rdmd/virtual_subjects'
    data_dir='/data/pySBI/rdmd/virtual_subjects_no_mutual_inh'
    subj_ids=range(20)
    virtual_subjects=[]
    for subj_idx in subj_ids:
         virtual_subjects.append(read_subject_data(data_dir, subj_idx,
                                                ['control','depolarizing','hyperpolarizing']))
    analyze_choice_hysteresis(virtual_subjects, output_dir=data_dir)
    analyze_accuracy_rt(virtual_subjects, output_dir=data_dir)
    analyze_bias(virtual_subjects, output_dir=data_dir)
    plt.show()
