#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
##----- Job shop SimPy - Use simpy.env.step  ------
# * Author: Colin, Lee
# * Date: Nov 5th, 2021
##-------------------------------------------------
#
import os
import sys
import logging
if '..' not in sys.path:
    sys.path.append('..')

import time
import simpy
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import utils.dispatch_logic as dp_logic

from collections               import defaultdict
from matplotlib.animation      import FuncAnimation
from utils.GanttPlot           import Gantt
from utils.action_map          import ACTION_MAP

import config_djss as config
seed = config.SEED #999
np.random.seed(seed)

INFINITY   = float('inf')
OPTIMAL_L  = config.OPT_MAKESPAN
DIM_ACTION = config.DIM_ACTION

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
plt.set_loglevel('WARNING') 
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logging.info('\n* Start Simulate')

#entity
class Order:
    def __init__(self, id, routing, prc_time, rls_time):
        # *prc_time: process time, *rls_time: release time, *intvl_arr: arrival_interval
        self.id        = id
        self.routing   = routing
        self.prc_time  = prc_time
        self.rls_time  = rls_time
        # self.intvl_arr = intvl_arr
        self.arr_time  = rls_time
        self.wait_time = 0

        self.progress = 0

    def __str__(self):
        return f'Order #{self.id}'


#resource in factory
class Source:
    def __init__(self, fac, expo_inter_arvl):
        # reference
        self.fac = fac
        # attribute
        self.expo_inter_arvl = expo_inter_arvl
        self.expo_proc_time  = self.fac.expo_proc_time
        # self.order_info  = fac.order_info
        self.orders_list = []
        # statistics
        self.num_generated = 0
        # switch for delivery
        self.fac.batch_arrived = False

    def connect_to(self, dict_queues={}):
        #reference
        self.env    	  = self.fac.env
        self.output_ports = dict_queues

    def _send_to_port(self, order):
        # available to deliver
        if self.fac.batch_arrived == True:
            output_ports    = set()
            orders_for_port = defaultdict(list)
            for order in self.orders_list:
            # lookup the target queue of routing for each order 
                target = int(order.routing[order.progress])
                # open the correlated output port
                output_port = self.output_ports[target]
                output_ports.add(output_port)
                # prepare the order for the output port to deliver
                orders_for_port[output_port].append(order)

            # deliver the order for opened ports
            for port in output_ports:
                # send order to queue
                port.order_arrive(orders_for_port[port])
            # reset the batch of orders
            self.orders_list = []

    def _generate_order(self):
        #get the data of orders
        # self.order_info = self.order_info.sort_values(by='release_time')
        num_order       = self.fac.num_job

        inter_arvl = np.random.exponential(self.expo_inter_arvl)
        # for num in range(num_order):
        while True:
            ## Delay ## wait for inter-arrival time
            yield self.env.timeout(inter_arvl)

            # create an instance of Order
            id        = self.num_generated
            num_op    = np.random.randint(2, self.fac.num_machine + 1)
            routing   = np.arange(self.fac.num_machine)
            # prc_time  = [np.random.exponential(self.expo_proc_time) for _ in routing]
            prc_time  = [np.random.uniform(low=3, high=8) for _ in routing]
            np.random.shuffle(routing)
            counter = 0
            for _ in routing[num_op:]:
                routing[num_op + counter]  = -1
                prc_time[num_op + counter] = -1
                counter                   += 1
            rls_time = self.env.now
            order    = Order(id, routing, prc_time, rls_time)
			###############################
			# RL #
            if id < self.fac.num_job:
                self.fac.tb_machine_no[id] = routing
                self.fac.tb_proc_times[id] = np.array(prc_time)
			    ###############################
                # job_per_channel = self.fac.job_per_channel
                # indx    = order.id % job_per_channel
                # channel = int(order.id / job_per_channel)
                # self.fac.tb_machine_no[channel][indx] = routing
                # self.fac.tb_proc_times[channel][indx] = np.array(prc_time)
			###############################
            # update statistics
            self.num_generated += 1
            # log for tracing
            if self.fac.log:
                print(f"    ({self.env.now}) {order} released")
            # collection of batch for items
            self.orders_list.append(order)
            # check if batch arrival complete
            next_inter_arvl   = np.random.exponential(self.expo_inter_arvl)
            generated_a_batch = (self.num_generated == num_order)
            if next_inter_arvl != 0 or generated_a_batch:
                self.fac.batch_arrived = True

            # When the flow item is released, select which port need to open
            self._send_to_port(order)

            inter_arvl = next_inter_arvl


class Dispatcher:
    def __init__(self, fac, default_rule):
        # reference
        self.fac     = fac
        self.env     = fac.env
        # attributes
        self.dp_rule = default_rule
		# a dict {listener: callbackMethod}
        self.listeners = dict()

    def __str__(self):
        return f'dipatcher with {self.dp_rule} rule'

#################
#    TO DO:     
#################
    def _on_dispatching(self, machine):
        # mark as a breakpoint for observe state and choose action
        logging.debug('---- breakpoint ----')
        ###########################################
        self.fac.decision_epoch = True
        ###########################################
		# note-->  listeners = {listener: callbackMethod}
        callback                = getattr(machine.input_port, 'pop_order')     
        self.listeners[machine] = callback

        return True #None

#################
#    TO DO:     
#################
    def reply(self, mesg):
        # listening for a mesg
        logging.debug('and resume\n')
        # execute the action
        for listener, callback in self.listeners.items():
            callback(mesg)
		# clear all the listening event
        self.listeners = dict()
    
    def dispatch_for(self, machine):
		# make sure arrival event occur first
        yield self.env.timeout(0)

        queue               = machine.input_port
        mulit_jobs_in_queue = (len(queue.space) > 1)

        # check if queue size is greater than one
        if not mulit_jobs_in_queue:
            queue.pop_order(self.fac.default_rule)
        else:
            # check if onDispatching trigger event is null
            if self._on_dispatching(machine) is None:
                queue.pop_order(self.fac.default_rule)


class Queue:
    def __init__(self, fac, id):
        #reference
        self.fac     = fac
        self.env     = self.fac.env
        #attribute
        self.id         = id
        self.space      = []
        self.dspch_rule = None

        self.waiting_for_dspch = False

    def __str__(self):
        return f'queue #{self.id}'

    def connect_to(self, input_ports, output_port):
        #reference
        self.input_ports = input_ports
        self.output_port = output_port

    def order_arrive(self, *orders):
        # change the data format for batch of items
        if type(orders[0]) == type([]):
            orders = orders[0]
        # insert flow item into list
        for order in orders:
            self.space.append(order)
            # log for order arrival
            if self.fac.log:
                print(f"    ({self.env.now}) {self} arrive with {order} - {order.progress+1} progress")
            # udpate statistic
            order.arr_time = self.env.now
        # to check if the output port available and open it for item delivery
        self._send_to_port()

    def _send_to_port(self):
        # check if output_port is available and content of queue is not null
        machine           = self.output_port
        machine_idle      = (machine.status == 'idle')
        waiting_in_space  = (len(self.space) > 0)
        waiting_for_dspch = self.waiting_for_dspch
        # need to dipatch for queue
        if machine_idle and waiting_in_space and not waiting_for_dspch:
            # self.fac.dispatcher.dispatch_for(self)
            self.env.process(self.fac.dispatcher.dispatch_for(machine))
            self.waiting_for_dspch = True

    def select_order_by(self, mesg):
        dp_rule = mesg
        # get order from space
        order = dp_logic.get_order_from(self.space, dp_rule)
        return order

    def pop_order(self, mesg):
        # # get order from space
        # order = dp_logic.get_order_from(self.space, dp_rule)
        order = self.select_order_by(mesg)
        # remove order form queue
        self.space.remove(order)
        # downstream machine recieve a job 
        self.output_port.recieve_order(order)

        self.waiting_for_dspch = False


class Machine:
    def __init__(self, fac, id, num = 1):
        self.fac = fac
        self.id  = id
        self.num = num
        #reference
        self.env    = self.fac.env
        #attributes
        self.status = "idle"
        #statistics
        self.using_time = 0

    def __str__(self):
        return f'machine #{self.id}'
    
    def connect_to(self, input_port, ouput_ports):
        #reference
        self.input_port   = input_port
        self.output_ports = ouput_ports # dictionary about {'queues', 'sink'}

    def recieve_order(self, order):
        # log for process order
        if self.fac.log:
            print(f"    ({self.env.now}) {self}  start processing {order} - {order.progress+1} progress")
        # change status
        self.status = order

        ########################################################
        ########################################################
        if order.arr_time is not None:
            order.wait_time += self.env.now - order.arr_time
            order.arr_time = None

        ########################################################
        ########################################################
        # TO DO:     
        #   - update the table of assignment (which operation has assigned)
        if order.id < self.fac.num_job:
            # self.fac.tb_asgn_status[order.id][order.progress] = 1
            self.fac.tb_proc_status[order.id][order.progress] = 1
            ########################################################
            # job_per_channel = self.fac.job_per_channel
            # channel = int(order.id / job_per_channel)
            # indx    = order.id % job_per_channel
            # self.fac.tb_asgn_status[channel][indx][order.progress] = 1
        ########################################################
        ########################################################

        # start processing (delay process time)
        self.process = self.env.process(self._process_order())

    def _process_order(self):
        # processing order for prc_time mins
        order         = self.status
        mean_prc_time = order.prc_time[order.progress]
        # prc_time      = np.random.normal(mean_prc_time, 2)
        # while prc_time < 0:
        #     prc_time = np.random.normal(mean_prc_time, 2)
        prc_time = order.prc_time[order.progress]
        # [Gantt plot preparing] udate the start/finish processing time of machine
        self.fac.gantt_plot.update_gantt(self.id, \
                                         self.env.now, \
                                         prc_time, \
                                         order.id)
        ## Delay ## delay processing 
        yield self.env.timeout(prc_time)
		########################################################
        ########################################################
        # TO DO:     
        #   - update the table of process status(progress matrix)
        if order.id < self.fac.num_job:
            # self.fac.tb_proc_status[order.id][order.progress] = 1
            self.fac.tb_proc_status[order.id][order.progress] = 2
            ########################################################
            # job_per_channel = self.fac.job_per_channel
            # channel = int(order.id / job_per_channel)
            # indx    = order.id % job_per_channel
            # self.fac.tb_proc_status[channel][indx][order.progress] = 1
        ########################################################
        ########################################################
        # log after processing
        if self.fac.log:
            print(f"    ({self.env.now}) {self} finish processing {order} - {order.progress+1} progress")
        # change order status
        order.progress += 1
        # When the flow item is released, select which port need to open
        self._send_to_port(order)
        # change self status 
        self.using_time += prc_time
        self.status = 'idle'
        
        #################
        #    TO DO:     
        #################
        ## get next order in queue or done
		# check if reach terminal condition
        terminal_is_true = self.fac.terminal.triggered
        if terminal_is_true:
            self.fac.decision_epoch = True

        # send mesg to the queue, let it push an order out
        self.input_port._send_to_port()

    def _send_to_port(self, order):
        # check if order finished its routing
        if order.progress < len(order.routing):
            target = int(order.routing[order.progress])
            if target < 0:
            # store order to sink
                output_port = self.output_ports['sink']
                output_port.complete_order(order)
            else:
            # send order to next route
                output_port = self.output_ports['queues'][target]
                output_port.order_arrive(order)
        else:
            # store order to sink
            output_port = self.output_ports['sink']
            output_port.complete_order(order)


class Sink:
    def __init__(self, fac):
        # reference
        self.fac   = fac
        self.env   = self.fac.env
        # attribure
        self.space = []
        # statistics
        self.order_statistic = pd.DataFrame(columns = [
                                            "id", \
                                            "release_time", \
                                            "complete_time", \
                                            "flow_time"
                                            ])

    def complete_order(self, order):
        # update statistics
        self.space.append(order)
        self.update_order_statistic(order)
        # update factory statistic
        self.fac.throughput += 1

        # ternimal condition
        # num_order = self.fac.num_job #self.fac.order_info.shape[0]
        num_order = self.fac.terminal_order_num
        if self.fac.throughput >= num_order:
            self.fac.terminal.succeed()
            self.fac.makespan = self.env.now
            self.fac.mean_flow_time = self.order_statistic['flow_time'][self.fac.warmup_job:].mean()

    def update_order_statistic(self, order):
        # data to record in table
        id            = order.id
        rls_time      = order.rls_time
        complete_time = self.env.now
        flow_time     = complete_time - rls_time
        # update the global table
        self.order_statistic.loc[id] = \
        [id, rls_time, complete_time, flow_time]


#factory
class Factory:
    def __init__(self, jssp_config, default_rule, util, log=False):
        self.log            = log
        # system config
        self.jssp_config  = jssp_config
        self.num_machine  = 4#6 #jssp_config.num_machine
        self.num_job      = 500 #1000 #jssp_config.num_job

        self.warmup_job   = 100 #200 #1000
        self.terminal_order_num = 300

        self.level_load      = util #0.8 #0.9
        self.num_op          = 3#6
        self.avg_prc_time    = 5.5
        self.expo_proc_time  = self.avg_prc_time #18
        # self.expo_inter_arvl = 18 * (1/ self.level_load)
        self.expo_inter_arvl = (self.avg_prc_time * self.num_op) / (self.num_machine * self.level_load)

        self.default_rule = default_rule
        # self.order_info   = pd.read_excel(file_name)
        # self.opt_makespan = opt_makespan

        # statistics
        self.throughput     = 0
        self.current_util   = 0
        self.last_util      = 0
        self.makespan       = None
        self.mean_flow_time = None

        ##################### RL ##########################
        self.dim_actions      = 10 #DIM_ACTION
        # self.dim_observations = (4, self.num_job, self.num_machine)
        self.dim_observations = (3, self.num_job, self.num_machine)
        # self.dim_observations = (4*10, int(self.num_job/10), self.num_machine)
        # self.dim_observations = (int(4*5), int(self.num_job/5), self.num_machine)

        from gym import spaces
        self.action_space = spaces.Discrete(self.dim_actions)
        ###################################################

    def build(self):
        ## build factory ##
        self.env           = simpy.Environment()
        self.source        = Source(self, self.expo_inter_arvl)
        self.dispatcher    = Dispatcher(self, self.default_rule)
        self.sink          = Sink(self)
        self.dict_queues   = {}
        self.dict_machines = {}
        for id in range(self.num_machine):
            self.dict_queues[id]   = Queue(self, id)
            self.dict_machines[id] = Machine(self, id)

        ##  make connection ## 
        # source
        output_ports = self.dict_queues
        self.source.connect_to(output_ports)
        # queue
        for id, queue in self.dict_queues.items():
            input_ports = {
                           'source': self.source, 
                           'machines': self.dict_machines
                          }
            output_port = self.dict_machines[id]
            queue.connect_to(input_ports, output_port)
        # machine
        for id, machine in self.dict_machines.items():
            input_port   = self.dict_queues[id]
            output_ports = {
                            'queues': self.dict_queues, 
                            'sink': self.sink
                           }
            machine.connect_to(input_port, output_ports)

        # [Gantt]
        self.gantt_plot = Gantt()

        ## terminal event ##
        self.terminal   = self.env.event()
        # initial process
        self.process    = self.env.process(self.source._generate_order())
		# next event algo --> simpy.env.step
        self.next_event = getattr(self.env, 'step')

		#################
		#    TO DO:     
		#################
        # [RL]
        from utils.action_map import ACTION_MAP
        self.action_map = ACTION_MAP

        # [RL] attributes for the Environment of RL
        self.dim_actions      = 10 #DIM_ACTION
        # self.dim_observations = (4, self.num_job, self.num_machine)
        self.dim_observations = (3, self.num_job, self.num_machine)
        # self.dim_observations = (int(4*5), int(self.num_job/5), self.num_machine)
        self.observations     = np.zeros(self.dim_observations)
        self.actions          = np.arange(self.dim_actions)

        # self.job_per_channel  = int(self.num_job/5)
        self.tb_machine_no  = np.zeros((self.num_job, self.num_machine))
        self.tb_proc_times  = np.zeros((self.num_job, self.num_machine))
        # self.tb_asgn_status = np.zeros((self.num_job, self.num_machine))
        self.tb_proc_status = np.zeros((self.num_job, self.num_machine))
        # self.tb_machine_no  = np.zeros((5, self.job_per_channel, self.num_machine))
        # self.tb_proc_times  = np.zeros((5, self.job_per_channel, self.num_machine))
        # self.tb_asgn_status = np.zeros((5, self.job_per_channel, self.num_machine))
        # self.tb_proc_status = np.zeros((5, self.job_per_channel, self.num_machine))

        # statistics
        self.throughput      = 0
        self.current_util    = 0
        self.last_util       = 0
        self.makespan        = None
        self.mean_flow_time  = None
        self.last_waste_time = 0
    
    def get_utilization(self):
        # compute average utiliztion of machines
        total_using_time = 0
        for _, machine in self.dict_machines.items():
            total_using_time += machine.using_time
        avg_using_time = total_using_time / self.num_machine

        if self.env.now:
            utilization = avg_using_time / self.env.now
        else:
            utilization = 0
        return utilization
        
#################
#    TO DO:     
#################
    def _get_state(self):
        # self.observations[0:5] = self.tb_machine_no
        # self.observations[5:10] = self.tb_proc_times
        # self.observations[10:15] = self.tb_asgn_status
        # self.observations[15:20] = self.tb_proc_status
        self.observations[0] = self.tb_machine_no
        self.observations[1] = self.tb_proc_times
        # self.observations[2] = self.tb_asgn_status
        # self.observations[3] = self.tb_proc_status
        self.observations[2] = self.tb_proc_status
        return self.observations.copy()

#################
#    TO DO:     
#################
    def _get_reward(self):
        last_waste_time  = self.last_waste_time

        total_queue_time = 0
        for _, queue in self.dict_queues.items():
            for order in queue.space:
                if order.arr_time is not None:
                    total_queue_time += (self.env.now - order.arr_time)
                else:
                    total_queue_time += order.wait_time

        total_mc_idle_time = 0
        for _, machine in self.dict_machines.items():
            total_mc_idle_time += (self.env.now - machine.using_time)

        avg_queue_time       = (total_queue_time/self.source.num_generated)
        avg_idle_time        = (total_queue_time/self.source.num_generated)
        waste_time           = avg_queue_time + avg_idle_time
        self.last_waste_time = waste_time
        if self.terminal.triggered:
            credit         = 1000
            lower_bound    = self.avg_prc_time * self.num_op
            mean_flow_time = self.sink.order_statistic['flow_time'][self.warmup_job:].mean()
            compare        = (mean_flow_time - lower_bound)
            return (last_waste_time - waste_time) + credit/compare + 10/(waste_time+.0000001)
        return (last_waste_time - waste_time) + 10/(waste_time+.0000001)

#################
#    TO DO:     
#################
    def reset(self):
        self._render_his = []
        self.build()

        self.decision_epoch = False
        while not self.decision_epoch:
            self.next_event()

        self.decision_epoch = False

        initial_state = self._get_state()
        return initial_state

#################
#    TO DO:     
#################
    def step(self, action):
        ## execute the action ##
        dp_rule = self.action_map[action]
        self.dispatcher.reply(dp_rule) # self.dispatcher.execute_dispatching(dp_rule)

        # print('\t=================== Resume =====================')

        self.decision_epoch = False
        while not self.decision_epoch:
            self.next_event()

        self.decision_epoch = False

        next_state = self._get_state()
        reward     = self._get_reward()
        done       = self.terminal.triggered
        info       = {}
        # if self.env._queue[0][0] > 308803:
        #     done = True
        return next_state, reward, done, info

#################
#    TO DO:     
#################
    def render(self, terminal=False, use_mode=False, motion_speed=0.000001):
        plt.set_loglevel('WARNING') 
        if use_mode:
            if len(self._render_his) % 2 == 0:
                plt.ioff()
                plt.close('all')
            # queues_status = {}
            # for id, queue in self.queues.items():
                # queues_status[id] = [str(order) for order in queue.space]
            # print(f'\n (time: {self.env.now}) - Status of queues:\n {queues_status}')
            plt.ion()
            plt.pause(motion_speed)
            fig = self.gantt_plot.draw_gantt(self.env.now)
            self._render_his.append(fig)

            if terminal:
                plt.ioff()
                trm_frame = [plt.close(fig) for fig in self._render_his[:-1]]
                plt.show()
        else:
            if terminal:
                plt.ion()
                fig = self.gantt_plot.draw_gantt(self.env.now)
                plt.pause(motion_speed)
                # plt.ioff()
                # plt.show()

    def close(self):
        plt.ioff()
        plt.close('all')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows'   , None)
    pd.set_option('display.width'      , 300)
    pd.set_option('max_colwidth'       , 100)

    # read problem config
    opt_makespan = 55
    file_name    = 'job_info.xlsx'
    file_dir     = os.getcwd() + '/input_data'
    file_path    = os.path.join(file_dir, file_name)

    default_rule = 'FIFO'
    
    # fac = Factory(6, 6, file_path, opt_makespan, rule, log=True)
    # fac.build()
    # fac.env.run(until=fac.terminal)

    # make a jss factory
    # fac = Factory(6, 6, file_path, opt_makespan, rule, log=True)
    # fac = Factory(file_path, rule, log=False)#True)
    
    from tqdm import tqdm
    # lst_makespans, lst_utils, lst_mean_flow_time = [], [], []
    df_result = pd.DataFrame(columns = [
                                       "util", \
                                       "rule", \
                                       "makespan", \
                                       "mean_flow_time", \
                                       "avg_utilization"
                                       ])
    counter = 0
    lst_util = [0.9] #0.6, 0.7, 0.8]
    for util in lst_util:
        fac = Factory(file_path, default_rule, util, log=False)#True)
        print(f'Util: {util}')
        for rule in range(10):
        # for rule in range(1):
            d_rule = ACTION_MAP[rule]
            print(f'Rule {rule}: {d_rule}')
            # for rep in tqdm(range(10)):
            for rep in tqdm(range(1)):
                state = fac.reset()
                done  = False
                while not done:
                    # action = 0
                    # print('\t------------------- state ----------------------')
                    # if fac.log: 
                    #     for id, que in fac.dict_queues.items():
                    #         mc = fac.dict_machines[id]
                    #         print(f"\t {mc}-->{mc.status}\n\t --{que}   {[str(order) for order in que.space]}\n")
                    # print('\t------------------------------------------------')
                    action = rule
                    # action = 2
                    # action = int(input(f' * ({fac.env.now}) Choose an action from [0 ~ 9]: '))
                    state_, reward, done, _ = fac.step(action)
                    state = state_

                    # fac.render(terminal=done, use_mode=True)
                mean_flow_time = fac.sink.order_statistic['flow_time'][fac.warmup_job:].mean()
                avg_utils      = fac.get_utilization()
                makespan       = fac.env.now
                # lst_mean_flow_time.append(mean_flow_time)
                # lst_makespans.append(fac.makespan)
                # lst_utils.append(avg_utils)
                # df_result.loc[rep + 10*rule] = \
                df_result.loc[counter] = \
                [util, rule, makespan, mean_flow_time, avg_utils]
                counter += 1
                # fig = fac.gantt_plot.draw_gantt(fac.env.now)
                # plt.show()
            # print('* Utilization: ')
            # print(lst_utils)
            # print()
            # print('* Makespan: ')
            # print(lst_makespans)
            # print('* Mean Flow Time: ')
            # print(lst_makespans)
            # print('\n===========')
            # print('* Result * ')
            # print('===========\n')
            # print(df_result)
        # df_result.to_excel("result_spt.xlsx")
        print(df_result)
