from datetime import time

import brian

from perceptchoice.model.monitor import WTAMonitor


brian.set_global_preferences(useweave=True,openmp=True,useweave_linear_diffeq =True,
                             gcc_options = ['-ffast-math','-march=native'],usecodegenweave = True,
                             usecodegenreset = True)
from brian.library.IF import exp_IF
from brian.library.synapses import exp_synapse, biexp_synapse
from brian.membrane_equations import Current, InjectedCurrent
from brian.network import Network, network_operation
from brian.neurongroup import NeuronGroup
from brian.stdunits import pF, nS, mV, ms, Hz, pA, nF
from brian.tools.parameters import Parameters
from brian.units import siemens, second
from brian.clock import defaultclock, Clock
from brian.directcontrol import PoissonGroup
from brian.equations import Equations
from brian.connections import DelayConnection, Connection
import numpy as np
from numpy.matlib import randn, rand


pyr_params=Parameters(
    C=0.5*nF,
    gL=25*nS,
    refractory=2*ms,
    w_nmda = 0.165 * nS,
    w_ampa_ext_correct = 2.1*nS,
    w_ampa_ext_incorrect = 0.0*nS,
    w_ampa_bak = 2.1*nS,
    w_ampa_rec = 0.05*nS,
    w_gaba = 1.3*nS,
)

inh_params=Parameters(
    C=0.2*nF,
    gL=20*nS,
    refractory=1*ms,
    w_nmda = 0.13 * nS,
    w_ampa_ext = 1.62*nS,
    w_ampa_bak = 1.63*nS,
    w_ampa_rec = 0.04*nS,
    w_gaba = 1.0*nS,
)

# Default parameters for a WTA network with multiple inhibitory populations
default_params=Parameters(
    # Neuron parameters
    C = 200 * pF,
    gL = 20 * nS,
    EL = -70 * mV,
    VT = -55 * mV,
    DeltaT = 3 * mV,
    Vr = -53 * mV,
    # Magnesium concentration
    Mg = 1,
    # Synapse parameters
    E_ampa = 0 * mV,
    E_nmda = 0 * mV,
    E_gaba_a = -70 * mV,
    tau_ampa = 2*ms,
    tau1_nmda = 2*ms,
    tau2_nmda = 100*ms,
    tau_gaba_a = 5*ms,
    # Connection probabilities
    p_e_e=0.08,
    p_e_i=0.1,
    p_i_i=0.1,
    p_i_e=0.2,
    # Background firing rate
    background_freq=900*Hz,
    # Input variance
    input_var=4*Hz,
    # Input refresh rate
    refresh_rate=60.0*Hz,
    # Number of response options
    num_groups=2,
    # Total size of the network (excitatory and inhibitory cells)
    network_group_size=2000,
    background_input_size=2000,
    mu_0=40.0,
    # Proportion of pyramidal cells getting task-related input
    f=.15,
    task_input_resting_rate=1*Hz,
    # Response threshold
    resp_threshold=25
)
default_params.p_a=default_params.mu_0/100.0
default_params.p_b=default_params.p_a
default_params.task_input_size=int(default_params.network_group_size*.8*default_params.f)

simulation_params=Parameters(
    trial_duration=4*second,
    stim_start_time=1*second,
    stim_end_time=3*second,
    dt=0.5*ms,
    ntrials=1,
    p_dcs=0*pA,
    i_dcs=0*pA,
    dcs_start_time=0*second,
    dcs_end_time=0*second,
)

# WTA network class - extends Brian's NeuronGroup
class WTANetworkGroup(NeuronGroup):

    ### Constructor
    #       N = total number of neurons per input group
    #       num_groups = number of input groups
    #       params = network parameters
    #       background_input = background input source
    #       task_inputs = task input sources
    def __init__(self, params=default_params, pyr_params=pyr_params(), inh_params=inh_params(),
                 background_input=None, task_inputs=None, clock=defaultclock):
        self.params=params
        self.pyr_params=pyr_params
        self.inh_params=inh_params
        self.background_input=background_input
        self.task_inputs=task_inputs

        ## Set up equations

        # Exponential integrate-and-fire neuron
        eqs = exp_IF(params.C, params.gL, params.EL, params.VT, params.DeltaT)

        eqs += Equations('g_muscimol : nS')
        # AMPA conductance - recurrent input current
        eqs += exp_synapse('g_ampa_r', params.tau_ampa, siemens)
        eqs += Current('I_ampa_r=g_ampa_r*(E-vm): amp', E=params.E_ampa)

        # AMPA conductance - background input current
        eqs += exp_synapse('g_ampa_b', params.tau_ampa, siemens)
        eqs += Current('I_ampa_b=g_ampa_b*(E-vm): amp', E=params.E_ampa)

        # AMPA conductance - task input current
        eqs += exp_synapse('g_ampa_x', params.tau_ampa, siemens)
        eqs += Current('I_ampa_x=g_ampa_x*(E-vm): amp', E=params.E_ampa)

        # Voltage-dependent NMDA conductance
        eqs += biexp_synapse('g_nmda', params.tau1_nmda, params.tau2_nmda, siemens)
        eqs += Equations('g_V = 1/(1+(Mg/3.57)*exp(-0.062 *vm/mV)) : 1 ', Mg=params.Mg)
        eqs += Current('I_nmda=g_V*g_nmda*(E-vm): amp', E=params.E_nmda)

        # GABA-A conductance
        eqs += exp_synapse('g_gaba_a', params.tau_gaba_a, siemens)
        eqs += Current('I_gaba_a=g_gaba_a*(E-vm): amp', E=params.E_gaba_a)

        eqs +=InjectedCurrent('I_dcs: amp')

        NeuronGroup.__init__(self, params.network_group_size, model=eqs, threshold=-20*mV, refractory=1*ms,
            reset=params.Vr, compile=True, freeze=True, clock=clock)

        self.init_subpopulations()

        self.init_connectivity(clock)

    ## Initialize excitatory and inhibitory subpopulations
    def init_subpopulations(self):
        # Main excitatory subpopulation
        self.e_size=int(self.params.network_group_size*.8)
        self.group_e=self.subgroup(self.e_size)
        self.group_e.C=self.pyr_params.C
        self.group_e.gL=self.pyr_params.gL
        self.group_e._refractory_time=self.pyr_params.refractory

        # Main inhibitory subpopulation
        self.i_size=int(self.params.network_group_size*.2)
        self.group_i=self.subgroup(self.i_size)
        self.group_i.C=self.inh_params.C
        self.group_i.gL=self.inh_params.gL
        self.group_i._refractory_time=self.inh_params.refractory

        # Input-specific sub-subpopulations
        self.groups_e=[]
        for i in range(self.params.num_groups):
            subgroup_e=self.group_e.subgroup(int(self.params.f*self.e_size))
            self.groups_e.append(subgroup_e)
        self.ns_e=self.group_e.subgroup(self.e_size-(self.params.num_groups*int(self.params.f*self.e_size)))

        # Initialize state variables
        self.vm = self.params.EL+randn(self.params.network_group_size)*mV
        self.group_e.g_ampa_b = rand(self.e_size)*self.pyr_params.w_ampa_ext_correct*2.0
        self.group_e.g_nmda = rand(self.e_size)*self.pyr_params.w_nmda*2.0
        self.group_e.g_gaba_a = rand(self.e_size)*self.pyr_params.w_gaba*2.0
        self.group_i.g_ampa_r = rand(self.i_size)*self.inh_params.w_ampa_rec*2.0
        self.group_i.g_ampa_b = rand(self.i_size)*self.inh_params.w_ampa_bak*2.0
        self.group_i.g_nmda = rand(self.i_size)*self.inh_params.w_nmda*2.0
        self.group_i.g_gaba_a = rand(self.i_size)*self.inh_params.w_gaba*2.0


    ## Initialize network connectivity
    def init_connectivity(self, clock):
        self.connections={}

        # Iterate over input groups
        for i in range(self.params.num_groups):

            # E population - recurrent connections
            self.connections['e%d->e%d_ampa' % (i,i)]=init_connection(self.groups_e[i], self.groups_e[i],
                'g_ampa_r', self.pyr_params.w_ampa_rec, self.params.p_e_e, delay=.5*ms, allow_self_conn=False)
            self.connections['e%d->e%d_nmda' % (i,i)]=init_connection(self.groups_e[i], self.groups_e[i],
                'g_nmda', self.pyr_params.w_nmda, self.params.p_e_e, delay=.5*ms, allow_self_conn=False)

        # E -> I excitatory connections
        self.connections['e->i_ampa']=init_connection(self.group_e, self.group_i, 'g_ampa_r', self.inh_params.w_ampa_rec,
            self.params.p_e_i, delay=.5*ms)
        self.connections['e->i_nmda']=init_connection(self.group_e, self.group_i, 'g_nmda', self.inh_params.w_nmda,
            self.params.p_e_i, delay=.5*ms)

        # I -> E - inhibitory connections
        self.connections['i->e_gabaa']=init_connection(self.group_i, self.group_e, 'g_gaba_a', self.pyr_params.w_gaba,
            self.params.p_i_e, delay=.5*ms)

        # I population - recurrent connections
        self.connections['i->i_gabaa']=init_connection(self.group_i, self.group_i, 'g_gaba_a', self.inh_params.w_gaba,
            self.params.p_i_i, delay=.5*ms, allow_self_conn=False)

        if self.background_input is not None:
            # Background -> E+I population connections
            self.connections['b->ampa']=DelayConnection(self.background_input, self, 'g_ampa_b', delay=.5*ms)
            self.connections['b->ampa'][:,:]=0
            for i in xrange(len(self.background_input)):
                if i<self.e_size:
                    self.connections['b->ampa'][i,i]=self.pyr_params.w_ampa_bak
                else:
                    self.connections['b->ampa'][i,i]=self.inh_params.w_ampa_bak

        if self.task_inputs is not None:
            # Task input -> E population connections
            for i in range(self.params.num_groups):
                self.connections['t%d->e%d_ampa' % (i,i)]=DelayConnection(self.task_inputs[i], self.groups_e[i],
                    'g_ampa_x')
                self.connections['t%d->e%d_ampa' % (i,i)].connect_one_to_one(weight=self.pyr_params.w_ampa_ext_correct,
                    delay=.5*ms)
                self.connections['t%d->e%d_ampa' % (i,1-i)]=DelayConnection(self.task_inputs[i], self.groups_e[1-i],
                    'g_ampa_x')
                self.connections['t%d->e%d_ampa' % (i,1-i)].connect_one_to_one(weight=self.pyr_params.w_ampa_ext_incorrect,
                    delay=.5*ms)


class AccumulatorNetwork(WTANetworkGroup):
    def __init__(self, params=default_params, pyr_params=pyr_params(), inh_params=inh_params(), background_input=None,
                 task_inputs=None, clock=defaultclock):
        super(AccumulatorNetwork, self).__init__(params=params, pyr_params=pyr_params, inh_params=inh_params, background_input=background_input,
            task_inputs=task_inputs, clock=clock)

    def init_subpopulations(self):
        super(AccumulatorNetwork, self).init_subpopulations()

        # Input-specific sub-subpopulations
        self.groups_i=[]
        for i in range(self.params.num_groups):
            subgroup_i=self.group_i.subgroup(int(self.i_size*.5))
            self.groups_i.append(subgroup_i)

    def init_connectivity(self, clock):
        super(AccumulatorNetwork, self).init_connectivity(clock)

        del self.connections['e->i_ampa']
        del self.connections['e->i_nmda']
        del self.connections['i->e_gabaa']
        del self.connections['i->i_gabaa']

        self.connections['e_ns->i_ampa']=init_connection(self.ns_e, self.group_i, 'g_ampa_r', self.inh_params.w_ampa_rec,
            self.params.p_e_i, delay=.5*ms)
        self.connections['e_ns->i_nmda']=init_connection(self.ns_e, self.group_i, 'g_nmda', self.inh_params.w_nmda,
            self.params.p_e_i, delay=.5*ms)
        # I -> E - inhibitory connections
        self.connections['i->e_ns_gabaa']=init_connection(self.group_i, self.ns_e, 'g_gaba_a', self.pyr_params.w_gaba,
            self.params.p_i_e, delay=.5*ms)

        # Iterate over input groups
        for i in range(self.params.num_groups):

            # E -> I excitatory connections
            self.connections['e%d->i%d_ampa' % (i,i)]=init_connection(self.groups_e[i], self.groups_i[i], 'g_ampa_r', self.inh_params.w_ampa_rec,
                self.params.p_e_i, delay=.5*ms)
            self.connections['e%d->i%d_nmda' % (i,i)]=init_connection(self.groups_e[i], self.groups_i[i], 'g_nmda', self.inh_params.w_nmda,
                self.params.p_e_i, delay=.5*ms)


            # I -> E - inhibitory connections
            self.connections['i%d->e%d_gabaa' % (i,i)]=init_connection(self.groups_i[i], self.groups_e[i], 'g_gaba_a', self.pyr_params.w_gaba,
                self.params.p_i_e, delay=.5*ms)

            # I population - recurrent connections
            self.connections['i%d->i%d_gabaa' % (i,i)]=init_connection(self.groups_i[i], self.groups_i[i], 'g_gaba_a', self.inh_params.w_gaba,
                self.params.p_i_i, delay=.5*ms, allow_self_conn=False)


def run_wta(wta_params, input_freq, sim_params, pyr_params=pyr_params(), inh_params=inh_params(),
            output_file=None, save_summary_only=False, record_neuron_state=False, record_spikes=True, record_firing_rate=True,
            record_inputs=False, plot_output=False, report='text'):
    """
    Run WTA network
       wta_params = network parameters
       input_freq = mean firing rate of each input group
       output_file = output file to write to
       save_summary_only = whether or not to save all data or just summary data to file
       record_lfp = record LFP data if true
       record_voxel = record voxel data if true
       record_neuron_state = record neuron state data if true
       record_spikes = record spike data if true
       record_firing_rate = record network firing rates if true
       record_inputs = record input firing rates if true
       plot_output = plot outputs if true
    """

    start_time = time()

    simulation_clock=Clock(dt=sim_params.dt)
    input_update_clock=Clock(dt=1/(wta_params.refresh_rate/Hz)*second)

    background_input=PoissonGroup(wta_params.background_input_size, rates=wta_params.background_freq,
        clock=simulation_clock)
    task_inputs=[]
    for i in range(wta_params.num_groups):
        task_inputs.append(PoissonGroup(wta_params.task_input_size, rates=wta_params.task_input_resting_rate,
                                        clock=simulation_clock))

    # Create WTA network
    wta_network=WTANetworkGroup(params=wta_params, background_input=background_input, task_inputs=task_inputs,
        pyr_params=pyr_params, inh_params=inh_params, clock=simulation_clock)

    @network_operation(when='start', clock=input_update_clock)
    def set_task_inputs():
        for idx in range(len(task_inputs)):
            rate=wta_params.task_input_resting_rate
            if sim_params.stim_start_time<=simulation_clock.t<sim_params.stim_end_time:
                rate=input_freq[idx]*Hz+np.random.randn()*wta_params.input_var
                if rate<wta_params.task_input_resting_rate:
                    rate=wta_params.task_input_resting_rate
            task_inputs[idx]._S[0, :]=rate

    @network_operation(clock=simulation_clock)
    def inject_current():
        if simulation_clock.t>sim_params.dcs_start_time:
            wta_network.group_e.I_dcs=sim_params.p_dcs
            wta_network.group_i.I_dcs=sim_params.i_dcs

    # Create network monitor
    wta_monitor=WTAMonitor(wta_network, sim_params, record_neuron_state=record_neuron_state, record_spikes=record_spikes,
                           record_firing_rate=record_firing_rate, record_inputs=record_inputs,
                           save_summary_only=save_summary_only, clock=simulation_clock)

    # Create Brian network and reset clock
    net=Network(background_input, task_inputs,set_task_inputs, wta_network, wta_network.connections.values(),
                wta_monitor.monitors.values(), inject_current)
    print "Initialization time: %.2fs" % (time() - start_time)

    # Run simulation
    start_time = time()
    net.run(sim_params.trial_duration, report=report)
    print "Simulation time: %.2fs" % (time() - start_time)

    # Write output to file
    if output_file is not None:
        start_time = time()
        wta_monitor.write_output(input_freq, output_file)
        print 'Wrote output to %s' % output_file
        print "Write output time: %.2fs" % (time() - start_time)

    # Plot outputs
    if plot_output:
        wta_monitor.plot()

    return wta_monitor


def init_connection(pop1, pop2, target_name, weight, p, delay=None, allow_self_conn=True):
    """
    Initialize a connection between two populations
    pop1 = population sending projections
    pop2 = populations receiving projections
    target_name = name of synapse type to project to
    weight = weight of connection
    p = probability of connection between any two neurons
    delay = delay
    allow_self_conn = allow neuron to project to itself
    """
    if delay is not None:
        conn=DelayConnection(pop1, pop2, target_name, sparseness=p, weight=weight, delay=delay)
    else:
        conn=Connection(pop1, pop2, sparseness=p, weight=weight)

    # Remove self-connections
    if not allow_self_conn and len(pop1)==len(pop2):
        for j in xrange(len(pop1)):
            conn[j,j]=0.0
            conn[j,j]=0.0
            if delay is not None:
                conn.delay[j,j]=0.0
                conn.delay[j,j]=0.0
    return conn