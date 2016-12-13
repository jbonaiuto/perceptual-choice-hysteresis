from datetime import datetime
import os
from matplotlib.mlab import normpdf
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon, norm, mannwhitneyu
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from perceptchoice.experiment.subject_info import stim_conditions, read_isi_subjects, read_subjects, isi_conditions
from perceptchoice.utils import FitSigmoid, FitRT, FitWeibull

stim_colors={
    'ShamPreAnode': 'b',
    'Anode': 'r',
    'ShamPreCathode': 'b',
    'Cathode': 'g'
}
isi_colors={
    'low':'c',
    'high':'m'
}

def analyze_subject_accuracy_rt(subject):
    """
    Analyze the accuracy and RT of a single subject
    """
    # Map of condition - coherence - accuracy (correct or incorrect for each trial)
    condition_coherence_accuracy={}
    # Map of condition - coherence - RT
    condition_coherence_rt={}
    # Map of condition - coherence - mean RT difference with control
    condition_coherence_rt_diff={}
    # Map of condition - accuracy threshold
    condition_accuracy_thresh={}

    # Iterate through conditions
    for condition,trial_data in subject.session_data.iteritems():

        # Init accuracy, RT maps
        condition_coherence_accuracy[condition]={}
        condition_coherence_rt[condition]={}

        # For each trial
        for trial_idx in range(trial_data.shape[0]):

            # Get trial data
            coherence=trial_data[trial_idx,2]
            correct=trial_data[trial_idx,3]
            rt=trial_data[trial_idx,6]

            # Update accuracy
            if not coherence in condition_coherence_accuracy[condition]:
                condition_coherence_accuracy[condition][coherence]=[]
            condition_coherence_accuracy[condition][np.abs(coherence)].append(float(correct))

            # Update RT
            if not coherence in condition_coherence_rt[condition]:
                condition_coherence_rt[condition][coherence]=[]
            condition_coherence_rt[condition][np.abs(coherence)].append(rt)

        # Compute accuracy threshold
        coherences = sorted(condition_coherence_accuracy[condition].keys())
        accuracy=[]
        for coherence in coherences:
            accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
        acc_fit = FitWeibull(coherences, accuracy, guess=[0.0, 0.2], display=0)
        condition_accuracy_thresh[condition]=acc_fit.inverse(0.8)

    # Compute RT diff
    for stim_condition in ['Anode', 'Cathode']:
        condition_coherence_rt_diff[stim_condition]={}
        coherences=sorted(condition_coherence_rt[stim_condition].keys())
        for coherence in coherences:
            condition_coherence_rt_diff[stim_condition][coherence]=np.mean(condition_coherence_rt[stim_condition][coherence])-np.mean(condition_coherence_rt['ShamPre%s' % stim_condition][coherence])

    # Compute RT diff for sham conditions - allows to compare sham conditions to each other
    condition_coherence_rt_diff['Sham']={}
    coherences=sorted(condition_coherence_rt['ShamPreAnode'].keys())
    for coherence in coherences:
        condition_coherence_rt_diff['Sham'][coherence]=np.mean(condition_coherence_rt['ShamPreAnode'][coherence])-np.mean(condition_coherence_rt['ShamPreCathode'][coherence])

    return condition_coherence_accuracy, condition_coherence_rt, condition_coherence_rt_diff, condition_accuracy_thresh


def analyze_subject_isi_accuracy_rt(subject):
    """
    Analyze the accuracy and RT of a single subject
    """
    # Map of coherence - accuracy (correct or incorrect for each trial)
    isi_coherence_accuracy={
        'low':{},
        'high':{}
    }
    # Map of coherence - RT
    isi_coherence_rt={
        'low':{},
        'high':{}
    }
    # Map of condition - coherence - mean RT difference with control
    isi_coherence_rt_diff={
        'low':{},
        'high':{}
    }
    # Map of isi - accuracy threshold
    isi_accuracy_thresh={}

    trial_data=subject.session_data['control']
    mean_iti=np.mean(trial_data[1:,7])
    # For each trial
    for trial_idx in range(trial_data.shape[0]):
        isi='low'
        if trial_data[trial_idx,7]>mean_iti:
            isi='high'


        # Get trial data
        coherence=trial_data[trial_idx,2]
        correct=trial_data[trial_idx,3]
        rt=trial_data[trial_idx,6]

        # Update accuracy
        if not coherence in isi_coherence_accuracy[isi]:
            isi_coherence_accuracy[isi][coherence]=[]
        isi_coherence_accuracy[isi][np.abs(coherence)].append(float(correct))

        # Update RT
        if not coherence in isi_coherence_rt[isi]:
            isi_coherence_rt[isi][coherence]=[]
        isi_coherence_rt[isi][np.abs(coherence)].append(rt)

    # Compute accuracy threshold
    for isi in ['low','high']:
        coherences = sorted(isi_coherence_accuracy[isi].keys())
        accuracy=[]
        for coherence in coherences:
            accuracy.append(np.mean(isi_coherence_accuracy[isi][coherence]))
        acc_fit = FitWeibull(coherences, accuracy, guess=[0.0, 0.2], display=0)
        isi_accuracy_thresh[isi]=acc_fit.inverse(0.8)

    isi_coherence_rt_diff['low']={}
    coherences=sorted(isi_coherence_rt['low'].keys())
    for coherence in coherences:
        isi_coherence_rt_diff['low'][coherence]=np.mean(isi_coherence_rt['low'][coherence])-np.mean(isi_coherence_rt['high'][coherence])

    return isi_coherence_accuracy, isi_coherence_rt, isi_coherence_rt_diff, isi_accuracy_thresh



def analyze_isi_accuracy_rt(subjects, output_dir):
    """
    Analyze accuracy and RT of all subjects
    """
    isi_conditions=['low','high']
    # Map of condition - coherence - mean accuracy for each subject
    isi_coherence_accuracy={
        'low':{},
        'high':{}
    }
    # Map of condition - coherence - mean RT for each subject
    isi_coherence_rt={
        'low':{},
        'high':{}
    }
    # Map of condition - coherence - mean RT difference with control for each subject
    isi_coherence_rt_diff={
        'low':{},
        'high':{}
    }
    # Map of condition - accuracy threshold for each subject
    isi_accuracy_thresh={
        'low':[],
        'high':[]
    }
    # For each subject
    for subj_id in subjects:
        subject=subjects[subj_id]

        # Get subject accuracy, RT, RT diff, and accuracy threshold
        subj_isi_coherence_accuracy, subj_isi_coherence_rt, subj_isi_coherence_rt_diff, subj_isi_accuracy_thresh=analyze_subject_isi_accuracy_rt(subject)

        # Iterate over conditions
        for isi in isi_conditions:

            # Add accuray threshold to list for this condition
            isi_accuracy_thresh[isi].append(subj_isi_accuracy_thresh[isi])

            # Update accuracy with mean accuracy for this subject
            for coherence in subj_isi_coherence_accuracy[isi]:
                if not coherence in isi_coherence_accuracy[isi]:
                    isi_coherence_accuracy[isi][coherence]=[]
                isi_coherence_accuracy[isi][coherence].append(np.mean(subj_isi_coherence_accuracy[isi][coherence]))

            # Update RT with mean RT for this subject
            for coherence in subj_isi_coherence_rt[isi]:
                if not coherence in isi_coherence_rt[isi]:
                    isi_coherence_rt[isi][coherence]=[]
                isi_coherence_rt[isi][coherence].append(np.mean(subj_isi_coherence_rt[isi][coherence]))

        # Update RT difference
        for isi in subj_isi_coherence_rt_diff:
            for coherence in subj_isi_coherence_rt_diff[isi]:
                if not coherence in isi_coherence_rt_diff[isi]:
                    isi_coherence_rt_diff[isi][coherence]=[]
                isi_coherence_rt_diff[isi][coherence].append(subj_isi_coherence_rt_diff[isi][coherence])

    # Compute accuracy threshold spread for each condition
    thresh_std={}
    for isi in isi_conditions:
        thresh_std[isi]=np.std(isi_accuracy_thresh[isi])/np.sqrt(len(subjects))

    plot_choice_accuracy(isi_conditions, isi_colors, isi_coherence_accuracy, thresh_std=thresh_std)
    plot_choice_rt_scaled(isi_conditions, isi_colors, 'high', isi_coherence_rt)
    plot_choice_rt_diff(['low'], isi_colors, isi_coherence_rt_diff)

    # Write accuracy data to file
    out_file=file(os.path.join(output_dir,'isi_accuracy.csv'),'w')
    out_file.write('SubjID')
    for isi in isi_coherence_accuracy:
        for coherence in sorted(isi_coherence_accuracy[isi]):
            out_file.write(',%sCoherence%.4fAccuracy' % (isi,coherence))
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for isi in isi_coherence_accuracy:
            for coherence in sorted(isi_coherence_accuracy[isi]):
                out_file.write(',%.4f' % isi_coherence_accuracy[isi][coherence][subj_idx])
        out_file.write('\n')
    out_file.close()

    # Write RT data to file
    out_file=file(os.path.join(output_dir,'isi_rt.csv'),'w')
    out_file.write('SubjID')
    for isi in isi_coherence_rt:
        for coherence in sorted(isi_coherence_rt[isi]):
            out_file.write(',%sCoherence%.4fRT' % (isi,coherence))
    out_file.write('\n')
    for subj_idx,subj in enumerate(subjects):
        out_file.write('%d' % (subj_idx+1))
        for isi in isi_coherence_rt:
            for coherence in sorted(isi_coherence_rt[isi]):
                out_file.write(',%.4f' % isi_coherence_rt[isi][coherence][subj_idx])
        out_file.write('\n')
    out_file.close()


    # Run stats on accuracy threshold
    (W,p)=wilcoxon(isi_accuracy_thresh['low'],isi_accuracy_thresh['high'])
    print('Accuracy Threshold: W(%d)=%.4f, p=%.4f' % (len(subjects)-1, W, p))

    print('')

    # Run stats on RT difference
    rtdiff_results={
        'coh': {}, 'intercept':{}
    }
    coh=[]
    rt_diff=[]
    for coherence in isi_coherence_rt_diff['low']:
        for diff in isi_coherence_rt_diff['low'][coherence]:
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
        rtdiff_results[param]['x']=result.params[param]
        rtdiff_results[param]['t']=result.tvalues[param]
        rtdiff_results[param]['p']=result.pvalues[param]

    print('RT Diff')
    print('B0: x=%.4f, t=%.4f, p=%.4f' % (rtdiff_results['intercept']['x'], rtdiff_results['intercept']['t'], rtdiff_results['intercept']['p']))
    print('B1: x=%.4f, t=%.4f, p=%.4f' % (rtdiff_results['coh']['x'], rtdiff_results['coh']['t'], rtdiff_results['coh']['p']))


def analyze_isi_choice_hysteresis(subjects, output_dir):
    """
    Analyze choice hysteresis of all subjects
    """
    # Map of last response (L or R) - condition - coherence - average choice for each subject
    isi_coherence_choices={
        'L*': {
            'low':{},
            'high':{}
        },
        'R*': {
            'low':{},
            'high':{}
        }
    }
    # Map of last response (L or R) - condition - coherence - sigmoid offset of each subject
    isi_sigmoid_offsets={
        'L*': {
            'low':[],
            'high':[]
        },
        'R*': {
            'low':[],
            'high':[]
        }
    }
    # Map of logistic parameter (a1 or a2) - condition - parameter value for each subject
    isi_logistic_params={
        'a1': {
            'low':[],
            'high':[]
        },
        'a2': {
            'low':[],
            'high':[]
        }
    }

    # For each subject
    for subj_id in subjects:
        subject=subjects[subj_id]
        # Get choices, sigmoid offsets, logistic params
        subj_isi_coherence_choices, subj_isi_sigmoid_offsets, subj_isi_logistic_params=analyze_subject_isi_choice_hysteresis(subject)

        # For each condition
        for isi in isi_conditions:

            # Update sigmoid offsets
            isi_sigmoid_offsets['L*'][isi].append(subj_isi_sigmoid_offsets['L*'][isi])
            isi_sigmoid_offsets['R*'][isi].append(subj_isi_sigmoid_offsets['R*'][isi])

            # Update logistic params
            isi_logistic_params['a1'][isi].append(subj_isi_logistic_params['a1'][isi])
            isi_logistic_params['a2'][isi].append(subj_isi_logistic_params['a2'][isi])

            # Update L* choices
            for coherence in subj_isi_coherence_choices['L*'][isi]:
                if not coherence in isi_coherence_choices['L*'][isi]:
                    isi_coherence_choices['L*'][isi][coherence]=[]
                isi_coherence_choices['L*'][isi][coherence].append(np.mean(subj_isi_coherence_choices['L*'][isi][coherence]))

            # Update R* choices
            for coherence in subj_isi_coherence_choices['R*'][isi]:
                if not coherence in isi_coherence_choices['R*'][isi]:
                    isi_coherence_choices['R*'][isi][coherence]=[]
                isi_coherence_choices['R*'][isi][coherence].append(np.mean(subj_isi_coherence_choices['R*'][isi][coherence]))

    # Plot histograms
    plot_indecision_hist(isi_conditions, isi_colors, isi_sigmoid_offsets, xlim=[-.2,.6], ylim=[0,35])

    plot_logistic_parameter_ratio(isi_conditions, isi_colors, isi_logistic_params, xlim=[-0.1,0.35],ylim=[0,35])

    # Output indecision point data (sigmoid offsets)
    output_file=file(os.path.join(output_dir,'isi_indecision.csv'),'w')
    output_file.write('SubjID')
    for direction in ['L*','R*']:
        for isi in isi_sigmoid_offsets[direction]:
            output_file.write(',%s%s' % (direction, isi))
    output_file.write('\n')
    for subj_idx, subj in enumerate(subjects):
        output_file.write('%d' % (subj_idx+1))
        for direction in ['L*','R*']:
            for isi in isi_sigmoid_offsets[direction]:
                output_file.write(',%.4f' % isi_sigmoid_offsets[direction][isi][subj_idx])
        output_file.write('\n')
    output_file.close()

    # Output logistic params
    output_file=file(os.path.join(output_dir,'isi_logistic.csv'),'w')
    output_file.write('SubjID')
    for param in ['a1','a2']:
        for isi in isi_logistic_params[param]:
            output_file.write(',%s%s' % (param, isi))
    output_file.write('\n')
    for subj_idx, subj in enumerate(subjects):
        output_file.write('%d' % (subj_idx+1))
        for param in ['a1','a2']:
            for isi in isi_logistic_params[param]:
                output_file.write(',%.4f' % isi_logistic_params[param][isi][subj_idx])
        output_file.write('\n')
    output_file.close()

    # Run stats on indecision point
    indec_results={}
    (indec_results['W'],indec_results['p'])=wilcoxon(np.array(isi_sigmoid_offsets['L*']['low'])-np.array(isi_sigmoid_offsets['R*']['low']),
        np.array(isi_sigmoid_offsets['L*']['high'])-np.array(isi_sigmoid_offsets['R*']['high']))
    print('Indecision Point: W=%.4f, p=%.4f' % (indec_results['W'], indec_results['p']))
    (indec_results['W'],indec_results['p'])=wilcoxon(np.array(isi_sigmoid_offsets['L*']['low'])-np.array(isi_sigmoid_offsets['R*']['low']))
    print('Indecision Point, low: W=%.4f, p=%.4f' % (indec_results['W'], indec_results['p']))
    (indec_results['W'],indec_results['p'])=wilcoxon(np.array(isi_sigmoid_offsets['L*']['high'])-np.array(isi_sigmoid_offsets['R*']['high']))
    print('Indecision Point, high: W=%.4f, p=%.4f' % (indec_results['W'], indec_results['p']))

    print('')

    # Run stats on logistic parameters
    log_results={}
    low_ratio=np.array(isi_logistic_params['a2']['low'])/np.array(isi_logistic_params['a1']['low'])
    high_ratio=np.array(isi_logistic_params['a2']['high'])/np.array(isi_logistic_params['a1']['high'])
    (log_results['W'],log_results['p'])=wilcoxon(low_ratio, high_ratio)
    print('Logistic Regression: W=%.4f, p=%.4f' % (log_results['W'],log_results['p']))
    (log_results['W'],log_results['p'])=wilcoxon(low_ratio)
    print('Logistic Regression, low: W=%.4f, p=%.4f' % (log_results['W'],log_results['p']))
    (log_results['W'],log_results['p'])=wilcoxon(high_ratio)
    print('Logistic Regression, high: W=%.4f, p=%.4f' % (log_results['W'],log_results['p']))
    print('')


def analyze_subject_isi_choice_hysteresis(subject):
    """
    Analyze choice hysteresis for a single subject
    """
    # Map of last response (L or R) - condition - coherence - choice
    isi_coherence_choices={
        'L*': {
            'low':{},
            'high':{}
        },
        'R*': {
            'low':{},
            'high':{}
        }
    }
    # Map of last response (L or R) - condition - sigmoid offset
    isi_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    # Map of logistic parameter (a1 or a2) - condition
    isi_logistic_params={
        'a1': {},
        'a2': {}
    }

    # Iterate over conditions
    trial_data=subject.session_data['control']
    mean_iti=np.mean(trial_data[1:,7])
    # For each trial
    for trial_idx in range(trial_data.shape[0]):
        isi='low'
        if trial_data[trial_idx,7]>mean_iti:
            isi='high'

        # Get coherence - negative coherences when direction is to the left
        coherence=trial_data[trial_idx,2]*trial_data[trial_idx,1]
        last_resp=trial_data[trial_idx,5]
        resp=trial_data[trial_idx,4]

        # Last response was left
        if last_resp<0:
            if not coherence in isi_coherence_choices['L*'][isi]:
                isi_coherence_choices['L*'][isi][coherence]=[]
                # Append 0 to list if left (-1) or 1 if right
            isi_coherence_choices['L*'][isi][coherence].append(np.max([0,resp]))
        # Last response was right
        elif last_resp>0:
            # List of rightward choices (0=left, 1=right)
            if not coherence in isi_coherence_choices['R*'][isi]:
                isi_coherence_choices['R*'][isi][coherence]=[]
                # Append 0 to list if left (-1) or 1 if right
            isi_coherence_choices['R*'][isi][coherence].append(np.max([0,resp]))

    # Compute sigmoid offsets
    for isi in ['low','high']:
        for dir in ['L*','R*']:
            choice_probs=[]
            full_coherences=[]
            for coherence in isi_coherence_choices[dir][isi]:
                choice_probs.append(np.mean(isi_coherence_choices[dir][isi][coherence]))
                full_coherences.append(coherence)
            acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
            isi_sigmoid_offsets[dir][isi]=acc_fit.inverse(0.5)

    low_trials=np.where(trial_data[:,7]<=mean_iti)[0]
    # Prepare data for logistic
    data=pd.DataFrame({
        'resp': np.clip(trial_data[low_trials,4],0,1),
        # negative coherences when direction is to the left
        'coh': trial_data[low_trials,2]*trial_data[low_trials,1],
        'last_resp': trial_data[low_trials,5]
    })
    # Fit intercept
    data['intercept']=1.0

    # Run logistic regression and get params
    data=data[np.isfinite(data['last_resp'])]
    logit = sm.Logit(data['resp'], data[['coh','last_resp','intercept']])
    result = logit.fit(disp=False)
    isi_logistic_params['a1']['low']=result.params['coh']
    isi_logistic_params['a2']['low']=result.params['last_resp']

    high_trials=np.where(trial_data[:,7]>mean_iti)[0]
    # Prepare data for logistic
    data=pd.DataFrame({
        'resp': np.clip(trial_data[high_trials,4],0,1),
        # negative coherences when direction is to the left
        'coh': trial_data[high_trials,2]*trial_data[high_trials,1],
        'last_resp': trial_data[high_trials,5]
    })
    # Fit intercept
    data['intercept']=1.0

    # Run logistic regression and get params
    data=data[np.isfinite(data['last_resp'])]
    logit = sm.Logit(data['resp'], data[['coh','last_resp','intercept']])
    result = logit.fit(disp=False)
    isi_logistic_params['a1']['high']=result.params['coh']
    isi_logistic_params['a2']['high']=result.params['last_resp']

    return isi_coherence_choices, isi_sigmoid_offsets, isi_logistic_params


def analyze_subject_choice_hysteresis(subject):
    """
    Analyze choice hysteresis for a single subject
    """
    # Map of last response (L or R) - condition - coherence - choice
    condition_coherence_choices={
        'L*': {},
        'R*': {}
    }
    # Map of last response (L or R) - condition - sigmoid offset
    condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    # Map of logistic parameter (a1 or a2) - condition
    condition_logistic_params={
        'a1': {},
        'a2': {}
    }

    # Iterate over conditions
    for condition,trial_data in subject.session_data.iteritems():
        # Dict of coherence levels
        condition_coherence_choices['L*'][condition]={}
        condition_coherence_choices['R*'][condition]={}

        # For each trial
        for trial_idx in range(trial_data.shape[0]):

            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]*trial_data[trial_idx,1]
            last_resp=trial_data[trial_idx,5]
            resp=trial_data[trial_idx,4]

            # Last response was left
            if last_resp<0:
                if not coherence in condition_coherence_choices['L*'][condition]:
                    condition_coherence_choices['L*'][condition][coherence]=[]
                # Append 0 to list if left (-1) or 1 if right
                condition_coherence_choices['L*'][condition][coherence].append(np.max([0,resp]))
            # Last response was right
            elif last_resp>0:
                # List of rightward choices (0=left, 1=right)
                if not coherence in condition_coherence_choices['R*'][condition]:
                    condition_coherence_choices['R*'][condition][coherence]=[]
                # Append 0 to list if left (-1) or 1 if right
                condition_coherence_choices['R*'][condition][coherence].append(np.max([0,resp]))

        # Compute sigmoid offsets
        for dir in ['L*','R*']:
            choice_probs=[]
            full_coherences=[]
            for coherence in condition_coherence_choices[dir][condition]:
                choice_probs.append(np.mean(condition_coherence_choices[dir][condition][coherence]))
                full_coherences.append(coherence)
            acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
            condition_sigmoid_offsets[dir][condition]=acc_fit.inverse(0.5)

        # Prepare data for logistic
        data=pd.DataFrame({
            'resp': np.clip(trial_data[1:,4],0,1),
            # negative coherences when direction is to the left
            'coh': trial_data[1:,2]*trial_data[1:,1],
            'last_resp': trial_data[1:,5]
        })
        # Fit intercept
        data['intercept']=1.0

        # Run logistic regression and get params
        logit = sm.Logit(data['resp'], data[['coh','last_resp','intercept']])
        result = logit.fit(disp=False)
        condition_logistic_params['a1'][condition]=result.params['coh']
        condition_logistic_params['a2'][condition]=result.params['last_resp']

    return condition_coherence_choices, condition_sigmoid_offsets, condition_logistic_params


def analyze_accuracy_rt(subjects, output_dir):
    """
    Analyze accuracy and RT of all subjects
    """
    # Map of condition - coherence - mean accuracy for each subject
    condition_coherence_accuracy={}
    # Map of condition - coherence - mean RT for each subject
    condition_coherence_rt={}
    # Map of condition - coherence - mean RT difference with control for each subject
    condition_coherence_rt_diff={}
    # Map of condition - accuracy threshold for each subject
    condition_accuracy_thresh={}
    # For each subject
    for subj_id in subjects:
        subject=subjects[subj_id]

        # Get subject accuracy, RT, RT diff, and accuracy threshold
        subj_condition_coherence_accuracy, subj_condition_coherence_rt, subj_condition_coherence_rt_diff,\
            subj_condition_accuracy_thresh=analyze_subject_accuracy_rt(subject)

        # Iterate over conditions
        for condition in stim_conditions:

            # Init maps
            if not condition in condition_coherence_accuracy:
                condition_coherence_accuracy[condition]={}
                condition_coherence_rt[condition]={}
                condition_accuracy_thresh[condition]=[]

            # Add accuray threshold to list for this condition
            condition_accuracy_thresh[condition].append(subj_condition_accuracy_thresh[condition])

            # Update accuracy with mean accuracy for this subject
            for coherence in subj_condition_coherence_accuracy[condition]:
                if not coherence in condition_coherence_accuracy[condition]:
                    condition_coherence_accuracy[condition][coherence]=[]
                condition_coherence_accuracy[condition][coherence].append(np.mean(subj_condition_coherence_accuracy[condition][coherence]))

            # Update RT with mean RT for this subject
            for coherence in subj_condition_coherence_rt[condition]:
                if not coherence in condition_coherence_rt[condition]:
                    condition_coherence_rt[condition][coherence]=[]
                condition_coherence_rt[condition][coherence].append(np.mean(subj_condition_coherence_rt[condition][coherence]))

        # Update RT difference
        for condition in subj_condition_coherence_rt_diff:
            if not condition in condition_coherence_rt_diff:
                condition_coherence_rt_diff[condition]={}
            for coherence in subj_condition_coherence_rt_diff[condition]:
                if not coherence in condition_coherence_rt_diff[condition]:
                    condition_coherence_rt_diff[condition][coherence]=[]
                condition_coherence_rt_diff[condition][coherence].append(subj_condition_coherence_rt_diff[condition][coherence])

    # Compute accuracy threshold spread for each condition
    thresh_std={}
    for condition in stim_conditions:
        thresh_std[condition]=np.std(condition_accuracy_thresh[condition])/np.sqrt(len(subjects))

    # Generate plots
    # One figure for each stimulation condition
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        sham_condition='ShamPre%s' % stim_condition
        conditions=[sham_condition, stim_condition]
        plot_choice_accuracy(conditions, stim_colors, condition_coherence_accuracy, thresh_std=thresh_std)

        plot_choice_rt_scaled(conditions, stim_colors, sham_condition, condition_coherence_rt)

    plot_choice_rt_diff(['Anode','Cathode'],stim_colors, condition_coherence_rt_diff)

    # Write accuracy data to file
    out_file=file(os.path.join(output_dir,'accuracy.csv'),'w')
    out_file.write('SubjID')
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

    # Write RT data to file
    out_file=file(os.path.join(output_dir,'rt.csv'),'w')
    out_file.write('SubjID')
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

    # Write RT diff data to file
    out_file=file(os.path.join(output_dir,'rt_diff.csv'),'w')
    out_file.write('SubjID')
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

    # Run stats on accuracy threshold
    thresh_results={
        'sham': {},
        'cathode': {},
        'anode': {},
        }
    (thresh_results['sham']['W'],thresh_results['sham']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreCathode'],condition_accuracy_thresh['ShamPreAnode'])
    (thresh_results['cathode']['W'],thresh_results['cathode']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreCathode'],condition_accuracy_thresh['Cathode'])
    (thresh_results['anode']['W'],thresh_results['anode']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreAnode'],condition_accuracy_thresh['Anode'])
    print('Accuracy Threshold')
    for condition, results in thresh_results.iteritems():
        N=len(subjects)
        print('%s: W(%d)=%.4f, p=%.4f' % (condition, N-1, results['W'], results['p']))

    print('')

    # Run stats on RT difference
    rtdiff_results={
        'sham': {'coh': {}, 'intercept':{}},
        'anode': {'coh': {}, 'intercept':{}},
        'cathode': {'coh': {}, 'intercept':{}}
    }
    for condition in ['Sham','Anode','Cathode']:
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
    print('RT Diff')
    for condition, results in rtdiff_results.iteritems():
        print('%s, B1: x=%.4f, t=%.4f, p=%.4f' % (condition, results['coh']['x'], results['coh']['t'],
                                                  results['coh']['p']))


def analyze_choice_hysteresis(subjects, output_dir):
    """
    Analyze choice hysteresis of all subjects
    """
    # Map of last response (L or R) - condition - coherence - average choice for each subject
    condition_coherence_choices={
        'L*': {},
        'R*': {}
    }
    # Map of last response (L or R) - condition - coherence - sigmoid offset of each subject
    condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    # Map of logistic parameter (a1 or a2) - condition - parameter value for each subject
    condition_logistic_params={
        'a1': {},
        'a2': {}
    }

    # For each subject
    for subj_id in subjects:
        subject=subjects[subj_id]
        # Get choices, sigmoid offsets, logistic params
        subj_condition_coherence_choices, subj_condition_sigmoid_offsets, subj_condition_logistic_params=analyze_subject_choice_hysteresis(subject)

        # For each condition
        for condition in stim_conditions:

            # Init maps
            if not condition in condition_coherence_choices['L*']:
                condition_coherence_choices['L*'][condition]={}
                condition_coherence_choices['R*'][condition]={}
                condition_sigmoid_offsets['L*'][condition]=[]
                condition_sigmoid_offsets['R*'][condition]=[]
                condition_logistic_params['a1'][condition]=[]
                condition_logistic_params['a2'][condition]=[]

            # Update sigmoid offsets
            condition_sigmoid_offsets['L*'][condition].append(subj_condition_sigmoid_offsets['L*'][condition])
            condition_sigmoid_offsets['R*'][condition].append(subj_condition_sigmoid_offsets['R*'][condition])

            # Update logistic params
            condition_logistic_params['a1'][condition].append(subj_condition_logistic_params['a1'][condition])
            condition_logistic_params['a2'][condition].append(subj_condition_logistic_params['a2'][condition])

            # Update L* choices
            for coherence in subj_condition_coherence_choices['L*'][condition]:
                if not coherence in condition_coherence_choices['L*'][condition]:
                    condition_coherence_choices['L*'][condition][coherence]=[]
                condition_coherence_choices['L*'][condition][coherence].append(np.mean(subj_condition_coherence_choices['L*'][condition][coherence]))

            # Update R* choices
            for coherence in subj_condition_coherence_choices['R*'][condition]:
                if not coherence in condition_coherence_choices['R*'][condition]:
                    condition_coherence_choices['R*'][condition][coherence]=[]
                condition_coherence_choices['R*'][condition][coherence].append(np.mean(subj_condition_coherence_choices['R*'][condition][coherence]))

    # Plot histograms
    # One plot for each stimulation condition
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        conditions=['ShamPre%s' % stim_condition, stim_condition]
        plot_indecision_hist(conditions, stim_colors, condition_sigmoid_offsets)
        plot_logistic_parameter_ratio(conditions, stim_colors, condition_logistic_params)

    # Output indecision point data (sigmoid offsets)
    output_file=file(os.path.join(output_dir,'indecision.csv'),'w')
    output_file.write('SubjID')
    for direction in ['L*','R*']:
        for condition in condition_sigmoid_offsets[direction]:
            output_file.write(',%s%s' % (direction, condition))
    output_file.write('\n')
    for subj_idx, subj in enumerate(subjects):
        output_file.write('%d' % (subj_idx+1))
        for direction in ['L*','R*']:
            for condition in condition_sigmoid_offsets[direction]:
                output_file.write(',%.4f' % condition_sigmoid_offsets[direction][condition][subj_idx])
        output_file.write('\n')
    output_file.close()

    # Output logistic params
    output_file=file(os.path.join(output_dir,'logistic.csv'),'w')
    output_file.write('SubjID')
    for param in ['a1','a2']:
        for condition in condition_logistic_params[param]:
            output_file.write(',%s%s' % (param, condition))
    output_file.write('\n')
    for subj_idx, subj in enumerate(subjects):
        output_file.write('%d' % (subj_idx+1))
        for param in ['a1','a2']:
            for condition in condition_logistic_params[param]:
                output_file.write(',%.4f' % condition_logistic_params[param][condition][subj_idx])
        output_file.write('\n')
    output_file.close()

    # Run stats on indecision point
    indec_results={
        'sham': {},
        'anode': {},
        'cathode': {},
        'sham_anode': {},
        'sham_cathode': {},
        }
    (indec_results['sham']['W'],indec_results['sham']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['ShamPreCathode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreCathode']),
        np.array(condition_sigmoid_offsets['L*']['ShamPreAnode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreAnode']))
    (indec_results['cathode']['W'],indec_results['cathode']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['ShamPreCathode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreCathode']),
        np.array(condition_sigmoid_offsets['L*']['Cathode'])-np.array(condition_sigmoid_offsets['R*']['Cathode']))
    (indec_results['anode']['W'],indec_results['anode']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['ShamPreAnode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreAnode']),
        np.array(condition_sigmoid_offsets['L*']['Anode'])-np.array(condition_sigmoid_offsets['R*']['Anode']))
    (indec_results['sham_anode']['W'],indec_results['sham_anode']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['ShamPreAnode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreAnode']))
    (indec_results['sham_cathode']['W'],indec_results['sham_cathode']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['ShamPreCathode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreCathode']))
    print('Indecision Points')
    for condition, results in indec_results.iteritems():
        print('%s: W=%.4f, p=%.4f' % (condition, results['W'], results['p']))

    print('')

    # Run stats on logistic parameters
    log_results={
        'sham': {},
        'anode': {},
        'cathode': {},
        'sham_anode': {},
        'sham_cathode': {},
        }
    sham_anode_ratio=np.array(condition_logistic_params['a2']['ShamPreAnode'])/np.array(condition_logistic_params['a1']['ShamPreAnode'])
    sham_cathode_ratio=np.array(condition_logistic_params['a2']['ShamPreCathode'])/np.array(condition_logistic_params['a1']['ShamPreCathode'])
    (log_results['sham']['W'],log_results['sham']['p'])=wilcoxon(sham_anode_ratio, sham_cathode_ratio)
    (log_results['sham_anode']['W'],log_results['sham_anode']['p'])=wilcoxon(sham_anode_ratio)
    (log_results['sham_cathode']['W'],log_results['sham_cathode']['p'])=wilcoxon(sham_cathode_ratio)

    anode_ratio=np.array(condition_logistic_params['a2']['Anode'])/np.array(condition_logistic_params['a1']['Anode'])
    (log_results['anode']['W'],log_results['anode']['p'])=wilcoxon(sham_anode_ratio, anode_ratio)

    cathode_ratio=np.array(condition_logistic_params['a2']['Cathode'])/np.array(condition_logistic_params['a1']['Cathode'])
    (log_results['cathode']['W'],log_results['cathode']['p'])=wilcoxon(sham_cathode_ratio, cathode_ratio)

    print('Logistic Regression')
    for condition, results in log_results.iteritems():
        print('%s: W=%.4f, p=%.4f' % (condition, results['W'],results['p']))
    print('')



def plot_choice_accuracy(plot_conditions, plot_colors, condition_coherence_accuracy, thresh_std=None):
    """
    Plot choice accuracy over coherence level for each condition
    colors = map (condition, color)
    condition_coherence_accuracy - map of condition - coherence - mean accuracy for each subject
    thresh_std = accuracy threshold spread for each condition
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot stimulation condition and preceding sham condition
    for condition in plot_conditions:
        # Fit accuracy to Weibull
        coherences = sorted(condition_coherence_accuracy[condition].keys())
        mean_accuracy = [np.mean(condition_coherence_accuracy[condition][coherence]) for coherence in coherences]
        stderr_accuracy = [np.std(condition_coherence_accuracy[condition][coherence])/np.sqrt(len(condition_coherence_accuracy[condition][coherence])) for coherence in coherences]
        acc_fit = FitWeibull(coherences, mean_accuracy, guess=[0.0, 0.2], display=0)

        # Plot with error
        smoothInt = np.arange(.01, 1.0, 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        ax.semilogx(smoothInt, smoothResp, plot_colors[condition], label=condition)
        ax.errorbar(coherences, mean_accuracy, yerr=stderr_accuracy, fmt='o%s' % plot_colors[condition])

        # Plot threshold and spread
        thresh=acc_fit.inverse(0.8)
        ax.plot([thresh,thresh],[0.5,1],'--%s' % plot_colors[condition])
        if thresh_std is not None:
            rect=Rectangle((thresh-.5*thresh_std[condition],0.5),thresh_std[condition], .5, alpha=0.25,
                facecolor=plot_colors[condition], edgecolor='none')
        ax.add_patch(rect)
    ax.set_xlim([0.01,1.0])
    ax.set_ylim([0.5,1.0])
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% Correct')



def plot_choice_rt_scaled(plot_conditions, plot_colors, control_condition, condition_coherence_rt):
    """
    Plot RT over coherence level for each condition - scaled to min and max during sham
    colors = map (condition, color)
    condition_coherence_rt - map of condition - coherence - mean RT for each subject
    """
    # One figure for each stimulation condition

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Compute scale based on min and max RT during corresponding sham conditoin
    control_coherences=sorted(condition_coherence_rt[control_condition].keys())
    control_mean_rt = [np.mean(condition_coherence_rt[control_condition][x]) for x in control_coherences]
    scale=1/(np.max(control_mean_rt)-np.min(control_mean_rt))

    # Plot stimulation condition and preceding sham condition
    for condition in plot_conditions:
        # Scale and fit RT
        coherences = sorted(condition_coherence_rt[condition].keys())
        mean_rt = [scale*(np.mean(condition_coherence_rt[condition][coherence])-np.min(control_mean_rt)) for coherence in coherences]
        stderr_rt = [scale*np.std(condition_coherence_rt[condition][coherence])/np.sqrt(len(condition_coherence_rt[condition][coherence])) for coherence in coherences]
        rt_fit = FitRT(coherences, mean_rt, guess=[1,1,1], display=0)

        # Plot with error
        smoothInt = np.arange(.01, 1.0, 0.001)
        smoothRT = rt_fit.eval(smoothInt)
        ax.semilogx(smoothInt, smoothRT, plot_colors[condition], label=condition)
        ax.errorbar(coherences, mean_rt, yerr=stderr_rt, fmt='o%s' % plot_colors[condition])
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT')
    ax.set_xlim([0.01,1])
    ax.set_ylim([-0.2, 1.6])


def plot_choice_rt_diff(plot_conditions, plot_colors, condition_coherence_rt_diff):
    """\
    Plot RT difference over coherence level for each stimulation condition
    colors = map (condition, color)
    condition_coherence_rt - map of condition - coherence - mean RT difference for each subject
    """
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    # Plot each stimulation condition
    for condition in plot_conditions:
        # Fit difference to a line
        coherences = np.array(sorted(condition_coherence_rt_diff[condition].keys()))
        mean_diff=np.array([np.mean(condition_coherence_rt_diff[condition][coherence]) for coherence in coherences])
        stderr_diff=[np.std(condition_coherence_rt_diff[condition][coherence])/np.sqrt(len(condition_coherence_rt_diff[condition][coherence])) for coherence in coherences]
        clf = LinearRegression()
        clf.fit(np.expand_dims(coherences,axis=1),np.expand_dims(mean_diff,axis=1))
        a = clf.coef_[0][0]
        b = clf.intercept_[0]
        r_sqr=clf.score(np.expand_dims(coherences,axis=1), np.expand_dims(mean_diff,axis=1))

        # Plot line with error
        ax.plot([np.min(coherences), np.max(coherences)], [a * np.min(coherences) + b, a * np.max(coherences) + b], '--%s' % plot_colors[condition],
            label='r^2=%.3f' % r_sqr)
        ax.errorbar(coherences, mean_diff, yerr=stderr_diff, fmt='o%s' % plot_colors[condition], label=condition)
    ax.legend(loc='best')
    ax.set_xlim([0,0.55])
    ax.set_ylim([-80,100])
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT Difference')


def plot_logistic_parameter_ratio(plot_conditions, plot_colors, condition_logistic_params, xlim=[-.1,.2], ylim=[0,35]):
    """
    Plot logistic parameter ratio (a2/a1) as a histogram
    colors = map (condition, color)
    condition_logistic_params = map of logistic parameter (a1 or a2) - condition - parameter value for each subject
    """
    # Plot limits and bin width
    xx=np.arange(xlim[0],xlim[1],0.001)
    binwidth=.02

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # For stimulation condition and preceding sham condition
    for condition in plot_conditions:
        # Compute ratio
        ratio=np.array(condition_logistic_params['a2'][condition]) / np.array(condition_logistic_params['a1'][condition])

        # Plot histogram
        bins=np.arange(min(ratio), max(ratio) + binwidth, binwidth)
        hist,edges=np.histogram(ratio, bins=bins)
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist/float(len(ratio))*100.0, color=plot_colors[condition], alpha=0.75, label=condition, width=binwidth)

        # Fit and plot Gaussian
        (mu, sigma) = norm.fit(ratio)
        y = normpdf(xx, mu, sigma)*binwidth*100.0
        ax.plot(xx, y, '%s--' % plot_colors[condition], linewidth=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    ax.set_xlabel('a2/a1')
    ax.set_ylabel('% subjects')


def plot_indecision_hist(plot_conditions, plot_colors, condition_sigmoid_offsets, xlim=[-.2,.4], ylim=[0,30]):
    """
    Plot indecision point histogram
    colors = map (condition, color)
    condition_sigmoid_offsets = map of last response (L or R) - condition - coherence - sigmoid offset of each subject
    """
    # Plot limits and bin width
    xx=np.arange(xlim[0],xlim[1],0.001)
    binwidth=0.03


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # For stimulation condition and preceding sham condition
    for condition in plot_conditions:
        # Compute difference (shift in indecision point)
        diff=np.array(condition_sigmoid_offsets['L*'][condition])-np.array(condition_sigmoid_offsets['R*'][condition])

        # Plot histogram
        bins=np.arange(min(diff), max(diff)+binwidth, binwidth)
        hist,edges=np.histogram(diff, bins=bins)
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist/float(len(diff))*100.0, color=plot_colors[condition], alpha=0.75, label=condition, width=binwidth)

        # Fit and plot Gaussian
        (mu, sigma) = norm.fit(diff)
        y = normpdf(xx, mu, sigma)*binwidth*100.0
        ax.plot(xx, y,'%s--' % plot_colors[condition], linewidth=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Left*-Right* Indifference')
    ax.set_ylabel('% subjects')
    ax.legend(loc='best')

def compare_stim_isi_hysteresis(stim_subjects, isi_subjects):
    # Map of last response (L or R) - condition - coherence - sigmoid offset of each subject
    stim_condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    # Map of logistic parameter (a1 or a2) - condition - parameter value for each subject
    stim_condition_logistic_params={
        'a1': {},
        'a2': {}
    }

    # For each subject
    for subj_id in stim_subjects:
        subject=stim_subjects[subj_id]
        # Get choices, sigmoid offsets, logistic params
        subj_condition_coherence_choices, subj_condition_sigmoid_offsets, subj_condition_logistic_params=analyze_subject_choice_hysteresis(subject)

        # For each condition
        for condition in stim_conditions:

            # Init maps
            if not condition in stim_condition_sigmoid_offsets['L*']:
                stim_condition_sigmoid_offsets['L*'][condition]=[]
                stim_condition_sigmoid_offsets['R*'][condition]=[]
                stim_condition_logistic_params['a1'][condition]=[]
                stim_condition_logistic_params['a2'][condition]=[]

            # Update sigmoid offsets
            stim_condition_sigmoid_offsets['L*'][condition].append(subj_condition_sigmoid_offsets['L*'][condition])
            stim_condition_sigmoid_offsets['R*'][condition].append(subj_condition_sigmoid_offsets['R*'][condition])

            # Update logistic params
            stim_condition_logistic_params['a1'][condition].append(subj_condition_logistic_params['a1'][condition])
            stim_condition_logistic_params['a2'][condition].append(subj_condition_logistic_params['a2'][condition])

    # Map of last response (L or R) - condition - coherence - sigmoid offset of each subject
    isi_sigmoid_offsets={
        'L*': {
            'low':[],
            'high':[]
        },
        'R*': {
            'low':[],
            'high':[]
        }
    }
    # Map of logistic parameter (a1 or a2) - condition - parameter value for each subject
    isi_logistic_params={
        'a1': {
            'low':[],
            'high':[]
        },
        'a2': {
            'low':[],
            'high':[]
        }
    }

    # For each subject
    for subj_id in isi_subjects:
        subject=isi_subjects[subj_id]
        # Get choices, sigmoid offsets, logistic params
        subj_isi_coherence_choices, subj_isi_sigmoid_offsets, subj_isi_logistic_params=analyze_subject_isi_choice_hysteresis(subject)

        # For each condition
        for isi in isi_conditions:

            # Update sigmoid offsets
            isi_sigmoid_offsets['L*'][isi].append(subj_isi_sigmoid_offsets['L*'][isi])
            isi_sigmoid_offsets['R*'][isi].append(subj_isi_sigmoid_offsets['R*'][isi])

            # Update logistic params
            isi_logistic_params['a1'][isi].append(subj_isi_logistic_params['a1'][isi])
            isi_logistic_params['a2'][isi].append(subj_isi_logistic_params['a2'][isi])

    indec_results={
        'sham_anode': {},
        'sham_cathode': {},
        }
    (indec_results['sham_anode']['U'],indec_results['sham_anode']['p'])=mannwhitneyu(np.array(stim_condition_sigmoid_offsets['L*']['ShamPreAnode'])-np.array(stim_condition_sigmoid_offsets['R*']['ShamPreAnode']),np.array(isi_sigmoid_offsets['L*']['low'])-np.array(isi_sigmoid_offsets['R*']['low']))
    (indec_results['sham_cathode']['U'],indec_results['sham_cathode']['p'])=mannwhitneyu(np.array(stim_condition_sigmoid_offsets['L*']['ShamPreCathode'])-np.array(stim_condition_sigmoid_offsets['R*']['ShamPreCathode']),np.array(isi_sigmoid_offsets['L*']['low'])-np.array(isi_sigmoid_offsets['R*']['low']))
    print('Indecision Points')
    for condition, results in indec_results.iteritems():
        print('%s: U=%.4f, p=%.4f' % (condition, results['U'], results['p']))
    print('')
    # Run stats on logistic parameters
    log_results={
        'sham_anode': {},
        'sham_cathode': {},
        }
    sham_anode_ratio=np.array(stim_condition_logistic_params['a2']['ShamPreAnode'])/np.array(stim_condition_logistic_params['a1']['ShamPreAnode'])
    sham_cathode_ratio=np.array(stim_condition_logistic_params['a2']['ShamPreCathode'])/np.array(stim_condition_logistic_params['a1']['ShamPreCathode'])
    isi_ratio=np.array(isi_logistic_params['a2']['low'])/np.array(isi_logistic_params['a1']['low'])
    (log_results['sham_anode']['U'],log_results['sham_anode']['p'])=mannwhitneyu(sham_anode_ratio,isi_ratio)
    (log_results['sham_cathode']['U'],log_results['sham_cathode']['p'])=mannwhitneyu(sham_cathode_ratio,isi_ratio)

    print('Logistic Regression')
    for condition, results in log_results.iteritems():
        print('%s: U=%.4f, p=%.4f' % (condition, results['U'],results['p']))

if __name__=='__main__':
    print('*** Main analysis ***')
    data_dir='../../../rdmd/data/stim'
    stim_subjects=read_subjects(data_dir)
    analyze_choice_hysteresis(stim_subjects, data_dir)
    analyze_accuracy_rt(stim_subjects, data_dir)

    print('\n*** ISI analysis ***')
    data_dir='../../../rdmd/data/isi'
    isi_subjects=read_isi_subjects(data_dir)
    analyze_isi_choice_hysteresis(isi_subjects, data_dir)
    analyze_isi_accuracy_rt(isi_subjects, data_dir)
    plt.show()

    print('\n*** Compare ISI analysis ***')
    compare_stim_isi_hysteresis(stim_subjects, isi_subjects)
