from brian import Clock, Hz, second, PoissonGroup, network_operation, pA, Network
import numpy as np

from perceptchoice.model.monitor import WTAMonitor
from perceptchoice.model.network import default_params, pyr_params, inh_params, simulation_params, WTANetworkGroup


class VirtualSubject:
    def __init__(self, subj_id, wta_params=default_params(), pyr_params=pyr_params(), inh_params=inh_params(),
                 sim_params=simulation_params(), network_class=WTANetworkGroup):
        self.subj_id = subj_id
        self.wta_params = wta_params
        self.pyr_params = pyr_params
        self.inh_params = inh_params
        self.sim_params = sim_params

        self.simulation_clock = Clock(dt=self.sim_params.dt)
        self.input_update_clock = Clock(dt=1 / (self.wta_params.refresh_rate / Hz) * second)

        self.background_input = PoissonGroup(self.wta_params.background_input_size,
            rates=self.wta_params.background_freq, clock=self.simulation_clock)
        self.task_inputs = []
        for i in range(self.wta_params.num_groups):
            self.task_inputs.append(PoissonGroup(self.wta_params.task_input_size,
                rates=self.wta_params.task_input_resting_rate, clock=self.simulation_clock))

        # Create WTA network
        self.wta_network = network_class(params=self.wta_params, background_input=self.background_input,
            task_inputs=self.task_inputs, pyr_params=self.pyr_params, inh_params=self.inh_params,
            clock=self.simulation_clock)


        # Create network monitor
        self.wta_monitor = WTAMonitor(self.wta_network, self.sim_params, record_neuron_state=False, record_spikes=False,
                                      record_firing_rate=True, record_inputs=True, save_summary_only=False,
                                      clock=self.simulation_clock)


        # Create Brian network and reset clock
        self.net = Network(self.background_input, self.task_inputs, self.wta_network,
            self.wta_network.connections.values(), self.wta_monitor.monitors.values())


    def run_trial(self, sim_params, input_freq):
        self.wta_monitor.sim_params=sim_params
        self.net.reinit(states=False)

        @network_operation(when='start', clock=self.input_update_clock)
        def set_task_inputs():
            for idx in range(len(self.task_inputs)):
                rate = self.wta_params.task_input_resting_rate
                if sim_params.stim_start_time <= self.simulation_clock.t < sim_params.stim_end_time:
                    rate = input_freq[idx] * Hz + np.random.randn() * self.wta_params.input_var
                    if rate < self.wta_params.task_input_resting_rate:
                        rate = self.wta_params.task_input_resting_rate
                self.task_inputs[idx]._S[0, :] = rate

        @network_operation(clock=self.simulation_clock)
        def inject_current():
            if sim_params.dcs_start_time < self.simulation_clock.t <= sim_params.dcs_end_time:
                self.wta_network.group_e.I_dcs = sim_params.p_dcs
                self.wta_network.group_i.I_dcs = sim_params.i_dcs
            else:
                self.wta_network.group_e.I_dcs = 0 * pA
                self.wta_network.group_i.I_dcs = 0 * pA

        self.net.remove(set_task_inputs, inject_current)

        self.net.add(set_task_inputs, inject_current)

        self.net.run(sim_params.trial_duration, report='text')
