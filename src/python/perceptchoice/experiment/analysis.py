from datetime import datetime
import os
from matplotlib.mlab import normpdf
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from perceptchoice.experiment.subject_info import conditions, read_subjects
from perceptchoice.utils import FitSigmoid, FitRT, FitWeibull

colors={
    'ShamPreAnode': 'b',
    'Anode': 'r',
    'ShamPreCathode': 'b',
    'Cathode': 'g'
}

def analyze_subject_accuracy_rt(subject):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    for condition,trial_data in subject.session_data.iteritems():
        condition_coherence_accuracy[condition]={}
        condition_coherence_rt[condition]={}
        # For each trial
        for trial_idx in range(trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]
            correct=trial_data[trial_idx,3]
            rt=trial_data[trial_idx,6]

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

    for stim_condition in ['Anode', 'Cathode']:
        condition_coherence_rt_diff[stim_condition]={}
        coherences=sorted(condition_coherence_rt[stim_condition].keys())
        for coherence in coherences:
            condition_coherence_rt_diff[stim_condition][coherence]=np.mean(condition_coherence_rt[stim_condition][coherence])-np.mean(condition_coherence_rt['ShamPre%s' % stim_condition][coherence])
    condition_coherence_rt_diff['Sham']={}
    coherences=sorted(condition_coherence_rt['ShamPreAnode'].keys())
    for coherence in coherences:
        condition_coherence_rt_diff['Sham'][coherence]=np.mean(condition_coherence_rt['ShamPreAnode'][coherence])-np.mean(condition_coherence_rt['ShamPreCathode'][coherence])

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
        'a1': {},
        'a2': {}
    }
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

        for dir in ['L*','R*']:
            choice_probs=[]
            full_coherences=[]
            for coherence in condition_coherence_choices[dir][condition]:
                choice_probs.append(np.mean(condition_coherence_choices[dir][condition][coherence]))
                full_coherences.append(coherence)
            acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
            condition_sigmoid_offsets[dir][condition]=acc_fit.inverse(0.5)

        data=pd.DataFrame({
            'resp': np.clip(trial_data[1:,4],0,1),
            'coh': trial_data[1:,2]*trial_data[1:,1],
            'last_resp': trial_data[1:,5]
        })
        data['intercept']=1.0

        logit = sm.Logit(data['resp'], data[['coh','last_resp','intercept']])
        result = logit.fit(disp=False)
        condition_logistic_params['a1'][condition]=result.params['coh']
        condition_logistic_params['a2'][condition]=result.params['last_resp']

    return condition_coherence_choices, condition_sigmoid_offsets, condition_logistic_params



def analyze_accuracy_rt(subjects, output_dir):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    thresh_std={}
    # For each subject
    for subj_id in subjects:
        subject=subjects[subj_id]

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

    for condition in conditions:
        thresh_std[condition]=np.std(condition_accuracy_thresh[condition])/np.sqrt(len(subjects))

    plot_choice_accuracy(colors, condition_coherence_accuracy, plot_err=True, thresh_std=thresh_std)

    plot_choice_rt_scaled(colors, condition_coherence_rt)

    plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=True)

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

    thresh_results={
        'sham': {},
        'cathode': {},
        'anode': {},
        }
    (thresh_results['sham']['W'],thresh_results['sham']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreCathode'],condition_accuracy_thresh['ShamPreAnode'])
    (thresh_results['cathode']['W'],thresh_results['cathode']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreCathode'],condition_accuracy_thresh['Cathode'])
    (thresh_results['anode']['W'],thresh_results['anode']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreAnode'],condition_accuracy_thresh['Anode'])

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

    print('Accuracy Threshold')
    for condition, results in thresh_results.iteritems():
        N=len(subjects)
        print('%s: W(%d)=%.4f, p=%.4f' % (condition, N-1, results['W'], results['p']))

    print('')

    print('RT Diff')
    for condition, results in rtdiff_results.iteritems():
        print('%s, B1: x=%.4f, t=%.4f, p=%.4f' % (condition, results['coh']['x'], results['coh']['t'],
                                                  results['coh']['p']))



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
        'a1': {},
        'a2': {}
    }

    # For each subject
    for subj_id in subjects:
        subject=subjects[subj_id]
        subj_condition_coherence_choices, subj_condition_sigmoid_offsets, subj_condition_logistic_params=analyze_subject_choice_hysteresis(subject)

        for condition in conditions:
            if not condition in condition_coherence_choices['L*']:
                condition_coherence_choices['L*'][condition]={}
                condition_coherence_choices['R*'][condition]={}
                condition_sigmoid_offsets['L*'][condition]=[]
                condition_sigmoid_offsets['R*'][condition]=[]
                condition_logistic_params['a1'][condition]=[]
                condition_logistic_params['a2'][condition]=[]

            condition_sigmoid_offsets['L*'][condition].append(subj_condition_sigmoid_offsets['L*'][condition])
            condition_sigmoid_offsets['R*'][condition].append(subj_condition_sigmoid_offsets['R*'][condition])

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

    plot_logistic_parameter_ratio(colors, condition_logistic_params)

    output_file=file(os.path.join(output_dir,'indifference.csv'),'w')
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

    print('Indecision Points')
    for condition, results in indec_results.iteritems():
        print('%s: W=%.4f, p=%.4f' % (condition, results['W'], results['p']))

    print('')

    print('Logistic Regression')
    for condition, results in log_results.iteritems():
        print('%s: W=%.4f, p=%.4f' % (condition, results['W'],results['p']))
    print('')



def plot_choice_accuracy(colors, condition_coherence_accuracy, plot_err=False, thresh_std=None):
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            coherences = sorted(condition_coherence_accuracy[condition].keys())
            mean_accuracy = []
            stderr_accuracy = []
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
                rect=Rectangle((thresh-.5*thresh_std[condition],0.5),thresh_std[condition], .5, alpha=0.25,
                    facecolor=colors[condition], edgecolor='none')
            ax.add_patch(rect)
        ax.set_xlim([0.01,1.0])
        ax.set_ylim([0.5,1.0])
        ax.legend(loc='best')
        ax.set_xlabel('Coherence')
        ax.set_ylabel('% Correct')



def plot_choice_rt_scaled(colors, condition_coherence_rt):

    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        control_coherences=sorted(condition_coherence_rt['ShamPre%s' % stim_condition].keys())
        control_mean_rt = [np.mean(condition_coherence_rt['ShamPre%s' % stim_condition][x]) for x in control_coherences]
        scale=1/(np.max(control_mean_rt)-np.min(control_mean_rt))
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            coherences = sorted(condition_coherence_rt[condition].keys())
            mean_rt = []
            stderr_rt = []
            for coherence in coherences:
                mean_rt.append(scale*(np.mean(condition_coherence_rt[condition][coherence])-np.min(control_mean_rt)))
                stderr_rt.append(scale*np.std(condition_coherence_rt[condition][coherence])/np.sqrt(len(condition_coherence_rt[condition][coherence])))
            rt_fit = FitRT(coherences, mean_rt, guess=[1,1,1], display=0)
            smoothInt = np.arange(.01, 1.0, 0.001)
            smoothRT = rt_fit.eval(smoothInt)
            ax.semilogx(smoothInt, smoothRT, colors[condition], label=condition)
            ax.errorbar(coherences, mean_rt, yerr=stderr_rt, fmt='o%s' % colors[condition])
        ax.legend(loc='best')
        ax.set_xlabel('Coherence')
        ax.set_ylabel('RT')
        ax.set_xlim([0.01,1])
        ax.set_ylim([-0.2, 1.6])


def plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=False):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for stim_condition in ['Anode', 'Cathode']:
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
        r_sqr=clf.score(np.expand_dims(coherences,axis=1), np.expand_dims(mean_diff,axis=1))
        ax.plot([np.min(coherences), np.max(coherences)], [a * np.min(coherences) + b, a * np.max(coherences) + b], '--%s' % colors[stim_condition],
            label='r^2=%.3f' % r_sqr)

        if plot_err:
            ax.errorbar(coherences, mean_diff, yerr=stderr_diff, fmt='o%s' % colors[stim_condition], label=stim_condition)
        else:
            ax.plot(coherences, mean_diff, 'o%s' % colors[stim_condition], label=stim_condition)
    ax.legend(loc='best')
    ax.set_xlim([0,0.55])
    ax.set_ylim([-80,100])
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT Difference')


def plot_logistic_parameter_ratio(colors, condition_logistic_params):
    lims=[-.1,.2]
    xx=np.arange(lims[0],lims[1],0.001)
    binwidth=.02
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            ratio=np.array(condition_logistic_params['a2'][condition]) / np.array(condition_logistic_params['a1'][condition])
            bins=np.arange(min(ratio), max(ratio) + binwidth, binwidth)
            hist,edges=np.histogram(ratio, bins=bins)
            center = (bins[:-1] + bins[1:]) / 2
            ax.bar(center, hist/float(len(ratio))*100.0, color=colors[condition], alpha=0.75, label=condition, width=binwidth)
            (mu, sigma) = norm.fit(ratio)
            y = normpdf(xx, mu, sigma)*binwidth*100.0
            ax.plot(xx, y, '%s--' % colors[condition], linewidth=2)
        ax.set_xlim(lims)
        ax.set_ylim([0,35])
        ax.legend(loc='best')
        ax.set_xlabel('a2/a1')
        ax.set_ylabel('% subjects')


def plot_indifference_hist(colors, condition_sigmoid_offsets):
    binwidth=0.03
    lims=[-.2,.4]
    xx=np.arange(lims[0],lims[1],0.001)
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
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


if __name__=='__main__':
    data_dir='../../../rdmd/data/stim'

    excluded_subjects=[]
    subjects=read_subjects(data_dir)
    print('N subjects=%d' % len(subjects))
    analyze_choice_hysteresis(subjects, data_dir)
    analyze_accuracy_rt(subjects, data_dir)
    plt.show()

