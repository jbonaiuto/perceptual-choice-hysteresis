import h5py
from matplotlib.patches import Rectangle
import numpy as np
from brian import MultiStateMonitor, PopulationRateMonitor, SpikeMonitor, raster_plot, ms, hertz, nS, nA, defaultclock
import matplotlib.pyplot as plt

from perceptchoice.model.utils import get_response_time, plot_network_firing_rates
from perceptchoice.utils import FitRT


# Collection of monitors for WTA network
class SessionMonitor():
    def __init__(self, network, sim_params, conv_window=10, record_firing_rates=False):
        self.sim_params=sim_params
        self.network_params=network.params
        self.pyr_params=network.pyr_params
        self.inh_params=network.inh_params
        # Accuracy convolution window size
        self.conv_window=conv_window
        self.trial_inputs=np.zeros((self.network_params.num_groups,sim_params.ntrials))
        self.trial_rt=np.zeros((1,sim_params.ntrials))
        self.trial_resp=np.zeros((1,sim_params.ntrials))
        self.trial_correct=np.zeros((1,sim_params.ntrials))
        self.record_firing_rates=record_firing_rates
        self.correct_avg = np.zeros((1, sim_params.ntrials))
        self.pop_rates={}
        if self.record_firing_rates:
            for i,group_e in enumerate(network.groups_e):
                self.pop_rates['excitatory_rate_%d' % i]=[]

                self.pop_rates['inhibitory_rate']=[]
        self.num_no_response=0

    def record_trial(self, trial_idx, inputs, correct_input, wta_net, wta_monitor):
        self.trial_inputs[:,trial_idx]=inputs

        e_rate_0 = wta_monitor.monitors['excitatory_rate_0'].smooth_rate(width= 5 * ms, filter = 'gaussian')
        e_rate_1 = wta_monitor.monitors['excitatory_rate_1'].smooth_rate(width= 5 * ms, filter = 'gaussian')
        i_rate = wta_monitor.monitors['inhibitory_rate'].smooth_rate(width= 5 * ms, filter = 'gaussian')

        if self.record_firing_rates:
            self.pop_rates['excitatory_rate_0'].append(e_rate_0)
            self.pop_rates['excitatory_rate_1'].append(e_rate_1)
            self.pop_rates['inhibitory_rate'].append(i_rate)

        rt, choice = get_response_time(np.array([e_rate_0, e_rate_1]), self.sim_params.stim_start_time,
            self.sim_params.stim_end_time, upper_threshold = self.network_params.resp_threshold,
            dt = self.sim_params.dt)

        correct = choice == correct_input
        if choice>-1:
            print 'response time = %.3f correct = %d' % (rt, int(correct))
        else:
            print 'no response!'
            self.num_no_response+=1
        self.trial_rt[0,trial_idx]=rt
        self.trial_resp[0,trial_idx]=choice
        self.trial_correct[0,trial_idx]=correct
        self.correct_avg[0,trial_idx] = (np.sum(self.trial_correct))/(trial_idx+1)

    def get_correct_ma(self):
        correct_ma = np.convolve(self.trial_correct[0, :], np.ones((self.conv_window,)) / self.conv_window, mode='valid')
        return correct_ma

    def get_perc_correct(self):
        resp_trials=np.where(self.trial_resp[0,:]>-1)[0]
        perc_correct=float(np.sum(self.trial_correct[0,resp_trials]))/float(len(resp_trials))
        return perc_correct

    def get_perc_correct_test(self):
        resp_trials = np.where(self.trial_resp[0,:]>-1)[0]
        resp_trials_test = [x for x in resp_trials if x >= self.sim_params.ntrials/2]
        perc_correct_test = float(np.sum(self.trial_correct[0,resp_trials_test]))/float(len(resp_trials_test))
        return perc_correct_test

    def get_perc_correct_training(self):
        resp_trials=np.where(self.trial_resp[0,:]>-1)[0]
        resp_trials_training = [y for y in resp_trials if y < self.sim_params.ntrials/2]
        perc_correct_training = float(np.sum(self.trial_correct[0,resp_trials_training]))/float(len(resp_trials_training))
        return perc_correct_training

    def plot_mean_firing_rates(self, trials, plt_title='Mean Firing Rates'):
        if self.record_firing_rates:
            mean_e_pop_rates=[]
            std_e_pop_rates=[]
            for i in range(self.network_params.num_groups):
                pop_rate_mat=np.array(self.pop_rates['excitatory_rate_%d' % i])
                mean_e_pop_rates.append(np.mean(pop_rate_mat[trials,:],axis=0))
                std_e_pop_rates.append(np.std(pop_rate_mat[trials,:],axis=0)/np.sqrt(len(trials)))
            mean_e_pop_rates=np.array(mean_e_pop_rates)
            std_e_pop_rates=np.array(std_e_pop_rates)
            pop_rate_mat=np.array(self.pop_rates['inhibitory_rate'])
            mean_i_pop_rate=np.mean(pop_rate_mat[trials,:],axis=0)
            std_i_pop_rate=np.std(pop_rate_mat[trials,:],axis=0)/np.sqrt(len(trials))
            plot_network_firing_rates(np.array(mean_e_pop_rates), self.sim_params, self.network_params,
                std_e_rates=std_e_pop_rates, i_rate=mean_i_pop_rate, std_i_rate=std_i_pop_rate, plt_title=plt_title)

    def plot_sorted_mean_firing_rates(self, trials, plt_title='Mean Firing Rates'):
        if self.record_firing_rates:
            chosen_pop_rates=[]
            unchosen_pop_rates=[]
            for trial_idx in trials:
                resp=self.trial_resp[0,trial_idx]
                if resp>-1:
                    chosen_pop_rates.append(self.pop_rates['excitatory_rate_%d' % resp][trial_idx])
                    unchosen_pop_rates.append(self.pop_rates['excitatory_rate_%d' % (1-resp)][trial_idx])
            if len(chosen_pop_rates)>1:
                chosen_pop_rates=np.array(chosen_pop_rates)
                unchosen_pop_rates=np.array(unchosen_pop_rates)
                mean_e_pop_rates=np.array([np.mean(chosen_pop_rates,axis=0), np.mean(unchosen_pop_rates,axis=0)])
                std_e_pop_rates=np.array([np.std(chosen_pop_rates,axis=0)/np.sqrt(len(trials)),
                                          np.std(unchosen_pop_rates,axis=0)/np.sqrt(len(trials))])
                plot_network_firing_rates(np.array(mean_e_pop_rates), self.sim_params, self.network_params,
                    std_e_rates=std_e_pop_rates, plt_title=plt_title, labels=['chosen','unchosen'])

    def plot_perc_missed(self):
        plt.figure()
        coherence_levels=self.get_coherence_levels()
        perc_missed=[]
        for coherence in coherence_levels:
            trials=self.get_coherence_trials(coherence)
            responded_trials=np.intersect1d(np.where(self.trial_resp[0,:]>-1)[0],trials)
            perc_missed.append((1.0-float(len(responded_trials))/float(len(trials)))*100.0)
        plt.plot(coherence_levels,perc_missed,'o')
        plt.xlabel('Coherence')
        plt.ylabel('% Missed')

    def plot_coherence_rt(self):
        coherence_levels=self.get_coherence_levels()
        mean_rt=[]
        std_rt=[]
        for coherence in coherence_levels:
            trials=self.get_coherence_trials(coherence)
            responded_trials=np.intersect1d(np.where(self.trial_resp[0,:]>-1)[0],trials)
            mean_rt.append(np.mean(self.trial_rt[0,responded_trials]))
            std_rt.append(np.std(self.trial_rt[0,responded_trials])/np.sqrt(len(responded_trials)))
        plt.figure()
        rt_fit = FitRT(coherence_levels, mean_rt, guess=[1,1,1], display=0)
        smoothInt = np.arange(min(coherence_levels), max(coherence_levels), 0.001)
        smoothRT = rt_fit.eval(smoothInt)
        plt.semilogx(smoothInt, smoothRT,'b')
        plt.errorbar(coherence_levels, mean_rt, yerr=std_rt, fmt='bo')
        plt.xlabel('Coherence')
        plt.ylabel('RT')

    def get_coherence_levels(self):
        trial_coherence_levels=[]
        for idx in range(self.sim_params.ntrials):
            coherence = np.abs((self.trial_inputs[0, idx] - self.network_params.mu_0) / (self.network_params.p_a * 100.0))
            trial_coherence_levels.append(float('%.3f' % coherence))
        return sorted(np.unique(trial_coherence_levels))

    def get_coherence_trials(self, coherence):
        trial_coherence_idx=[]
        for idx in range(self.sim_params.ntrials):
            trial_coherence = np.abs((self.trial_inputs[0, idx] - self.network_params.mu_0) / (self.network_params.p_a * 100.0))
            if ('%.3f' % trial_coherence)==('%.3f' % coherence):
                trial_coherence_idx.append(idx)

        return np.array(trial_coherence_idx)

    def plot(self):
        # Convolve accuracy
        correct_ma = self.get_correct_ma()

        if self.record_firing_rates:
            self.plot_mean_firing_rates(range(self.sim_params.ntrials))
            self.plot_sorted_mean_firing_rates(range(self.sim_params.ntrials))

            coherence_levels=self.get_coherence_levels()

            for coherence in coherence_levels:
                trials=self.get_coherence_trials(coherence)
                #self.plot_mean_firing_rates(trials, plt_title='Coherence=%.3f' % coherence)
                self.plot_sorted_mean_firing_rates(trials, plt_title='Coherence=%.3f' % coherence)

        self.plot_coherence_rt()
        self.plot_perc_missed()

        plt.figure()
        plt.plot(self.trial_correct[0,:])
        plt.xlabel('trial')
        plt.ylabel('correct choice = 1')

        plt.figure()
        plt.plot(self.correct_avg[0,:], label = 'average')
        plt.plot(correct_ma, label = 'moving avg')
        plt.legend(loc = 'best')
        plt.ylim(0,1)
        plt.xlabel('trial')
        plt.ylabel('accuracy')

        plt.figure()
        plt.plot(self.trial_rt[0,:])
        plt.ylim(0, 2000)
        plt.xlabel('trial')
        plt.ylabel('response time')

        plt.figure()
        plt.plot(self.trial_resp[0,:])
        plt.xlabel('trial')
        plt.ylabel('choice e0=0 e1=1')
        #plt.show()

    def write_output(self, output_file):

        f = h5py.File(output_file, 'w')

        # Write basic parameters

        f.attrs['conv_window']=self.conv_window

        f_sim_params=f.create_group('sim_params')
        for attr, value in self.sim_params.iteritems():
            f_sim_params.attrs[attr] = value

        f_network_params=f.create_group('network_params')
        for attr, value in self.network_params.iteritems():
            f_network_params.attrs[attr] = value

        f_pyr_params=f.create_group('pyr_params')
        for attr, value in self.pyr_params.iteritems():
            f_pyr_params.attrs[attr] = value

        f_inh_params=f.create_group('inh_params')
        for attr, value in self.inh_params.iteritems():
            f_inh_params.attrs[attr] = value

        f_behav=f.create_group('behavior')
        f_behav['num_no_response']=self.num_no_response
        f_behav['trial_rt']=self.trial_rt
        f_behav['trial_resp']=self.trial_resp
        f_behav['trial_correct']=self.trial_correct

        f_neur=f.create_group('neural')
        f_neur['trial_inputs']=self.trial_inputs

        f_rates=f_neur.create_group('firing_rates')
        if self.record_firing_rates:
            for trial_idx in range(self.sim_params.ntrials):
                f_trial=f_rates.create_group('trial_%d' % trial_idx)
                f_trial['inhibitory_rate']=self.pop_rates['inhibitory_rate'][trial_idx]
                for i in range(self.network_params.num_groups):
                    f_trial['excitatory_rate_%d' % i]=self.pop_rates['excitatory_rate_%d' % i][trial_idx]
        f.close()


class WTAMonitor():

    ## Constructor
    #       network = network to monitor
    #       record_neuron_state = record neuron state signals if true
    #       record_spikes = record spikes if true
    #       record_firing_rate = record firing rate if true
    #       record_inputs = record inputs if true
    def __init__(self, network, sim_params, record_neuron_state=False, record_spikes=True, record_firing_rate=True,
                 record_inputs=False, save_summary_only=False, clock=defaultclock):
        self.network_params=network.params
        self.pyr_params=network.pyr_params
        self.inh_params=network.inh_params
        self.sim_params=sim_params
        self.monitors={}
        self.save_summary_only=save_summary_only
        self.clock=clock
        self.record_neuron_state=record_neuron_state
        self.record_spikes=record_spikes
        self.record_firing_rate=record_firing_rate
        self.record_inputs=record_inputs
        self.save_summary_only=save_summary_only

        # Network monitor
        if self.record_neuron_state:
            self.record_idx=[]
            for i in range(self.network_params.num_groups):
                e_idx=i*int(.8*self.network_params.network_group_size/self.network_params.num_groups)
                self.record_idx.append(e_idx)
            i_idx=int(.8*self.network_params.network_group_size)
            self.record_idx.append(i_idx)
            self.monitors['network'] = MultiStateMonitor(network, vars=['vm','g_ampa_r','g_ampa_x','g_ampa_b',
                                                                        'g_gaba_a', 'g_nmda','I_ampa_r','I_ampa_x',
                                                                        'I_ampa_b','I_gaba_a','I_nmda'],
                record=self.record_idx, clock=clock)

        # Population rate monitors
        if self.record_firing_rate:
            for i,group_e in enumerate(network.groups_e):
                self.monitors['excitatory_rate_%d' % i]=PopulationRateMonitor(group_e)

            self.monitors['inhibitory_rate']=PopulationRateMonitor(network.group_i)

        # Input rate monitors
        if record_inputs:
            self.monitors['background_rate']=PopulationRateMonitor(network.background_input)
            for i,task_input in enumerate(network.task_inputs):
                self.monitors['task_rate_%d' % i]=PopulationRateMonitor(task_input)

        # Spike monitors
        if self.record_spikes:
            for i,group_e in enumerate(network.groups_e):
                self.monitors['excitatory_spike_%d' % i]=SpikeMonitor(group_e)

            self.monitors['inhibitory_spike']=SpikeMonitor(network.group_i)


    # Plot monitor data
    def plot(self):

        # Spike raster plots
        if self.record_spikes:
            num_plots=self.network_params.num_groups+1
            plt.figure()
            for i in range(self.network_params.num_groups):
                plt.subplot(num_plots,1,i+1)
                raster_plot(self.monitors['excitatory_spike_%d' % i],newfigure=False)
            plt.subplot(num_plots,1,num_plots)
            raster_plot(self.monitors['inhibitory_spike'],newfigure=False)

        # Network firing rate plots
        if self.record_firing_rate:

            e_rate_0=self.monitors['excitatory_rate_0'].smooth_rate(width=5*ms)/hertz
            e_rate_1=self.monitors['excitatory_rate_1'].smooth_rate(width=5*ms)/hertz
            #i_rate=self.monitors['inhibitory_rate'].smooth_rate(width=5*ms)/hertz
            #plot_network_firing_rates(np.array([e_rate_0, e_rate_1]), i_rate, self.sim_params, self.network_params)
            plot_network_firing_rates(np.array([e_rate_0, e_rate_1]), self.sim_params, self.network_params)

        # Input firing rate plots
        if self.record_inputs:
            plt.figure()
            ax= plt.subplot(111)
            max_rate=0
            task_rates=[]
            for i in range(self.network_params.num_groups):
                task_monitor=self.monitors['task_rate_%d' % i]
                task_rate=task_monitor.smooth_rate(width=5*ms,filter='gaussian')/hertz
                if np.max(task_rate)>max_rate:
                    max_rate=np.max(task_rate)
                task_rates.append(task_rate)

            rect=Rectangle((0,0),(self.sim_params.stim_end_time-self.sim_params.stim_start_time)/ms, max_rate+5,
                alpha=0.25, facecolor='yellow', edgecolor='none')
            ax.add_patch(rect)
            for i in range(self.network_params.num_groups):
                ax.plot((np.array(range(len(task_rates[i])))*self.sim_params.dt)/ms-self.sim_params.stim_start_time/ms, task_rates[i])
            plt.ylim(0,90)
            plt.ylabel('Firing rate (Hz)')
            plt.xlabel('Time (ms)')

        # Network state plots
        if self.record_neuron_state:
            network_monitor=self.monitors['network']
            max_conductances=[]
            for neuron_idx in self.record_idx:
                max_conductances.append(np.max(network_monitor['g_ampa_r'][neuron_idx]/nS))
                max_conductances.append(np.max(network_monitor['g_ampa_x'][neuron_idx]/nS))
                max_conductances.append(np.max(network_monitor['g_ampa_b'][neuron_idx]/nS))
                max_conductances.append(np.max(network_monitor['g_nmda'][neuron_idx]/nS))
                max_conductances.append(np.max(network_monitor['g_gaba_a'][neuron_idx]/nS))
            max_conductance=np.max(max_conductances)

            fig= plt.figure()
            for i in range(self.network_params.num_groups):
                neuron_idx=self.record_idx[i]
                ax= plt.subplot(int('%d1%d' % (self.network_params.num_groups+1,i+1)))
                plt.title('e%d' % i)
                ax.plot(network_monitor['g_ampa_r'].times/ms, network_monitor['g_ampa_r'][neuron_idx]/nS,
                    label='AMPA-recurrent')
                ax.plot(network_monitor['g_ampa_x'].times/ms, network_monitor['g_ampa_x'][neuron_idx]/nS,
                    label='AMPA-task')
                ax.plot(network_monitor['g_ampa_b'].times/ms, network_monitor['g_ampa_b'][neuron_idx]/nS,
                    label='AMPA-backgrnd')
                ax.plot(network_monitor['g_nmda'].times/ms, network_monitor['g_nmda'][neuron_idx]/nS,
                    label='NMDA')
                ax.plot(network_monitor['g_gaba_a'].times/ms, network_monitor['g_gaba_a'][neuron_idx]/nS,
                    label='GABA_A')
                plt.ylim(0,max_conductance)
                plt.xlabel('Time (ms)')
                plt.ylabel('Conductance (nS)')
                plt.legend()

            neuron_idx=self.record_idx[self.network_params.num_groups]
            ax= plt.subplot('%d1%d' % (self.network_params.num_groups+1,self.network_params.num_groups+1))
            plt.title('i')
            ax.plot(network_monitor['g_ampa_r'].times/ms, network_monitor['g_ampa_r'][neuron_idx]/nS,
                label='AMPA-recurrent')
            ax.plot(network_monitor['g_ampa_x'].times/ms, network_monitor['g_ampa_x'][neuron_idx]/nS,
                label='AMPA-task')
            ax.plot(network_monitor['g_ampa_b'].times/ms, network_monitor['g_ampa_b'][neuron_idx]/nS,
                label='AMPA-backgrnd')
            ax.plot(network_monitor['g_nmda'].times/ms, network_monitor['g_nmda'][neuron_idx]/nS,
                label='NMDA')
            ax.plot(network_monitor['g_gaba_a'].times/ms, network_monitor['g_gaba_a'][neuron_idx]/nS,
                label='GABA_A')
            plt.ylim(0,max_conductance)
            plt.xlabel('Time (ms)')
            plt.ylabel('Conductance (nS)')
            plt.legend()

            min_currents=[]
            max_currents=[]
            for neuron_idx in self.record_idx:
                max_currents.append(np.max(network_monitor['I_ampa_r'][neuron_idx]/nS))
                max_currents.append(np.max(network_monitor['I_ampa_x'][neuron_idx]/nS))
                max_currents.append(np.max(network_monitor['I_ampa_b'][neuron_idx]/nS))
                max_currents.append(np.max(network_monitor['I_nmda'][neuron_idx]/nS))
                max_currents.append(np.max(network_monitor['I_gaba_a'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_ampa_r'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_ampa_x'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_ampa_b'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_nmda'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_gaba_a'][neuron_idx]/nS))
            max_current=np.max(max_currents)
            min_current=np.min(min_currents)

            fig=plt.figure()
            for i in range(self.network_params.num_groups):
                ax=plt.subplot(int('%d1%d' % (self.network_params.num_groups+1,i+1)))
                neuron_idx=self.record_idx[i]
                plt.title('e%d' % i)
                ax.plot(network_monitor['I_ampa_r'].times/ms, network_monitor['I_ampa_r'][neuron_idx]/nA,
                    label='AMPA-recurrent')
                ax.plot(network_monitor['I_ampa_x'].times/ms, network_monitor['I_ampa_x'][neuron_idx]/nA,
                    label='AMPA-task')
                ax.plot(network_monitor['I_ampa_b'].times/ms, network_monitor['I_ampa_b'][neuron_idx]/nA,
                    label='AMPA-backgrnd')
                ax.plot(network_monitor['I_nmda'].times/ms, network_monitor['I_nmda'][neuron_idx]/nA,
                    label='NMDA')
                ax.plot(network_monitor['I_gaba_a'].times/ms, network_monitor['I_gaba_a'][neuron_idx]/nA,
                    label='GABA_A')
                plt.ylim(min_current,max_current)
                plt.xlabel('Time (ms)')
                plt.ylabel('Current (nA)')
                plt.legend()

            ax=plt.subplot(int('%d1%d' % (self.network_params.num_groups+1,self.network_params.num_groups+1)))
            neuron_idx=self.record_idx[self.network_params.num_groups]
            plt.title('i')
            ax.plot(network_monitor['I_ampa_r'].times/ms, network_monitor['I_ampa_r'][neuron_idx]/nA,
                label='AMPA-recurrent')
            ax.plot(network_monitor['I_ampa_x'].times/ms, network_monitor['I_ampa_x'][neuron_idx]/nA,
                label='AMPA-task')
            ax.plot(network_monitor['I_ampa_b'].times/ms, network_monitor['I_ampa_b'][neuron_idx]/nA,
                label='AMPA-backgrnd')
            ax.plot(network_monitor['I_nmda'].times/ms, network_monitor['I_nmda'][neuron_idx]/nA,
                label='NMDA')
            ax.plot(network_monitor['I_gaba_a'].times/ms, network_monitor['I_gaba_a'][neuron_idx]/nA,
                label='GABA_A')
            plt.ylim(min_current,max_current)
            plt.xlabel('Time (ms)')
            plt.ylabel('Current (nA)')
            plt.legend()


    ## Write monitor data to HDF5 file
    #       background_input_size = number of background inputs
    #       background_freq rate = background firing rate
    #       input_freq = input firing rates
    #       network_group_size = number of neurons per input group
    #       num_groups = number of input groups
    #       output_file = filename to write to
    #       record_firing_rate = write network firing rate data when true
    #       record_neuron_stae = write neuron state data when true
    #       record_spikes = write spike data when true
    #       record_voxel = write voxel data when true
    #       record_lfp = write LFP data when true
    #       record_inputs = write input firing rates when true
    #       stim_end_time = stimulation end time
    #       stim_start_time = stimulation start time
    #       task_input_size = number of neurons in each task input group
    #       trial_duration = duration of the trial
    #       voxel = voxel for network
    #       wta_monitor = network monitor
    #       wta_params = network parameters
    def write_output(self, input_freq, output_file):

        f = h5py.File(output_file, 'w')

        # Write basic parameters
        f.attrs['input_freq'] = input_freq

        f_sim_params=f.create_group('sim_params')
        for attr, value in self.sim_params.iteritems():
            f_sim_params.attrs[attr] = value

        f_network_params=f.create_group('network_params')
        for attr, value in self.network_params.iteritems():
            f_network_params.attrs[attr] = value

        f_pyr_params=f.create_group('pyr_params')
        for attr, value in self.pyr_params.iteritems():
            f_pyr_params.attrs[attr] = value

        f_inh_params=f.create_group('inh_params')
        for attr, value in self.inh_params.iteritems():
            f_inh_params.attrs[attr] = value

        if not self.save_summary_only:
            # Write neuron state data
            if self.record_neuron_state:
                f_state = f.create_group('neuron_state')
                f_state['g_ampa_r'] = self.monitors['network']['g_ampa_r'].values
                f_state['g_ampa_x'] = self.monitors['network']['g_ampa_x'].values
                f_state['g_ampa_b'] = self.monitors['network']['g_ampa_b'].values
                f_state['g_nmda'] = self.monitors['network']['g_nmda'].values
                f_state['g_gaba_a'] = self.monitors['network']['g_gaba_a'].values
                #f_state['g_gaba_b'] = self.monitors['network']['g_gaba_b'].values
                f_state['I_ampa_r'] = self.monitors['network']['I_ampa_r'].values
                f_state['I_ampa_x'] = self.monitors['network']['I_ampa_x'].values
                f_state['I_ampa_b'] = self.monitors['network']['I_ampa_b'].values
                f_state['I_nmda'] = self.monitors['network']['I_nmda'].values
                f_state['I_gaba_a'] = self.monitors['network']['I_gaba_a'].values
                #f_state['I_gaba_b'] = self.monitors['network']['I_gaba_b'].values
                f_state['vm'] = self.monitors['network']['vm'].values
                f_state['record_idx'] = np.array(self.record_idx)

            # Write network firing rate data
            if self.record_firing_rate:
                f_rates = f.create_group('firing_rates')
                e_rates = []
                for i in range(self.network_params.num_groups):
                    e_rates.append(self.monitors['excitatory_rate_%d' % i].smooth_rate(width=5 * ms, filter='gaussian'))
                f_rates['e_rates'] = np.array(e_rates)

                i_rates = [self.monitors['inhibitory_rate'].smooth_rate(width=5 * ms, filter='gaussian')]
                f_rates['i_rates'] = np.array(i_rates)

            # Write input firing rate data
            if self.record_inputs:
                back_rate=f.create_group('background_rate')
                back_rate['firing_rate']=self.monitors['background_rate'].smooth_rate(width=5*ms,filter='gaussian')
                task_rates=f.create_group('task_rates')
                t_rates=[]
                for i in range(self.network_params.num_groups):
                    t_rates.append(self.monitors['task_rate_%d' % i].smooth_rate(width=5*ms,filter='gaussian'))
                task_rates['firing_rates']=np.array(t_rates)

            # Write spike data
            if self.record_spikes:
                f_spikes = f.create_group('spikes')
                for idx in range(self.network_params.num_groups):
                    spike_monitor=self.monitors['excitatory_spike_%d' % idx]
                    if len(spike_monitor.spikes):
                        f_spikes['e.%d.spike_neurons' % idx] = np.array([s[0] for s in spike_monitor.spikes])
                        f_spikes['e.%d.spike_times' % idx] = np.array([s[1] for s in spike_monitor.spikes])

                spike_monitor=self.monitors['inhibitory_spike']
                if len(spike_monitor.spikes):
                    f_spikes['i.spike_neurons'] = np.array([s[0] for s in spike_monitor.spikes])
                    f_spikes['i.spike_times'] = np.array([s[1] for s in spike_monitor.spikes])

        else:
            f_summary=f.create_group('summary')
            endIdx=int(self.sim_params.stim_end_time/self.clock.dt)
            startIdx=endIdx-500
            e_mean_final=[]
            e_max=[]
            for idx in range(self.network_params.num_groups):
                rate_monitor=self.monitors['excitatory_rate_%d' % idx]
                e_rate=rate_monitor.smooth_rate(width=5*ms, filter='gaussian')
                e_mean_final.append(np.mean(e_rate[startIdx:endIdx]))
                e_max.append(np.max(e_rate))
            rate_monitor=self.monitors['inhibitory_rate']
            i_rate=rate_monitor.smooth_rate(width=5*ms, filter='gaussian')
            i_mean_final=[np.mean(i_rate[startIdx:endIdx])]
            i_max=[np.max(i_rate)]
            f_summary['e_mean']=np.array(e_mean_final)
            f_summary['e_max']=np.array(e_max)
            f_summary['i_mean']=np.array(i_mean_final)
            f_summary['i_max']=np.array(i_max)

        f.close()