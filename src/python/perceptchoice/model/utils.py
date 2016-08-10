from brian import second, ms, hertz
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure, subplot, ylim, legend, ylabel, title, xlabel
import numpy as np


def get_response_time(e_firing_rates, stim_start_time, stim_end_time, upper_threshold=60, threshold_diff=None, dt=.1*ms):
    rate_1=e_firing_rates[0]
    rate_2=e_firing_rates[1]
    times=np.array(range(len(rate_1)))*(dt/second)
    rt=None
    decision_idx=-1
    for idx,time in enumerate(times):
        time=time*second
        if stim_start_time < time < stim_end_time:
            if rt is None:
                if rate_1[idx]>=upper_threshold and (threshold_diff is None or rate_1[idx]-rate_2[idx]>=threshold_diff):
                    decision_idx=0
                    rt=(time-stim_start_time)/ms
                    break
                elif rate_2[idx]>=upper_threshold and (threshold_diff is None or rate_2[idx]-rate_1[idx]>=threshold_diff):
                    decision_idx=1
                    rt=(time-stim_start_time)/ms
                    break
    return rt,decision_idx

def plot_network_firing_rates(e_rates, sim_params, network_params, std_e_rates=None, i_rate=None, std_i_rate=None,
                              plt_title=None, labels=None, ax=None):
    rt, choice = get_response_time(e_rates, sim_params.stim_start_time, sim_params.stim_end_time,
                                   upper_threshold = network_params.resp_threshold, dt = sim_params.dt)

    if ax is None:
        figure()
    max_rates=[network_params.resp_threshold]
    if i_rate is not None:
        max_rates.append(np.max(i_rate[500:]))
    for i in range(network_params.num_groups):
        max_rates.append(np.max(e_rates[i,500:]))
    max_rate=np.max(max_rates)

    if i_rate is not None:
        ax=subplot(211)
    elif ax is None:
        ax=subplot(111)
    rect=Rectangle((0,0),(sim_params.stim_end_time-sim_params.stim_start_time)/ms, max_rate+5,
        alpha=0.25, facecolor='yellow', edgecolor='none')
    ax.add_patch(rect)

    for idx in range(network_params.num_groups):
        label='e %d' % idx
        if labels is not None:
            label=labels[idx]
        time_ticks=(np.array(range(e_rates.shape[1]))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        baseline,=ax.plot(time_ticks, e_rates[idx,:], label=label)
        if std_e_rates is not None:
            ax.fill_between(time_ticks, e_rates[idx,:]-std_e_rates[idx,:], e_rates[idx,:]+std_e_rates[idx,:], alpha=0.5,
                facecolor=baseline.get_color())
    ylim(0,max_rate+5)
    ax.plot([0-sim_params.stim_start_time/ms, (sim_params.trial_duration-sim_params.stim_start_time)/ms],
        [network_params.resp_threshold/hertz, network_params.resp_threshold/hertz], 'k--')
    ax.plot([rt,rt],[0, max_rate+5],'k--')
    legend(loc='best')
    ylabel('Firing rate (Hz)')
    if plt_title is not None:
        title(plt_title)

    if i_rate is not None:
        ax=subplot(212)
        rect=Rectangle((0,0),(sim_params.stim_end_time-sim_params.stim_start_time)/ms, max_rate+5,
            alpha=0.25, facecolor='yellow', edgecolor='none')
        ax.add_patch(rect)
        label='i'
        if labels is not None:
            label=labels[network_params.num_groups]
        time_ticks=(np.array(range(len(i_rate)))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        baseline,=ax.plot(time_ticks, i_rate, label=label)
        if std_i_rate is not None:
            ax.fill_between(time_ticks, i_rate-std_i_rate, i_rate+std_i_rate, alpha=0.5, facecolor=baseline.get_color())
        ylim(0,max_rate+5)
        ax.plot([rt,rt],[0, max_rate],'k--')
        ylabel('Firing rate (Hz)')
    xlabel('Time (ms)')


