import simpy
import pandas as pd
import random
import os
import functools

import numpy as np

from environment.RL_SimComponent import *


class Forming(object):
    def __init__(self):
        self.job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.weight = np.random.uniform(0, 5, len(self.job_type))
        self.machine_num = {'BH': 5}
        # job type 별 average process time
        self.p_ij = {'BH': np.random.uniform(10, 30, size=(len(self.job_type), self.machine_num['BH']))}
        self.p_j = {'BH': np.average(self.p_ij['BH'], axis=1)}
        self.process_list = ['BH', 'Sink']
        self.process_all = ['BH']
        self.priority = {'BH': [1, 2, 3, 4, 5]}
        self.part_num = 200
        self.IAT = np.random.exponential(scale=5, size=self.part_num)
        self.job_type_list = np.random.randint(low=0, high=10, size=self.part_num)
        # due date generating factor
        self.K = {'BH': 1}

        self.env, self.model, self.source = self._modeling()
        self.done = False

        self.mean_weighted_tardiness = 0
        self.make_span = 0

    def _modeling(self):
        # Make model
        env = simpy.Environment()
        model = dict()

        source = Source(env, self.IAT, self.weight, self.job_type, self.job_type_list, self.p_ij, model,
                        self.process_list,
                        self.machine_num,
                        self.part_num, self.K)

        for i in range(len(self.process_all) + 1):
            if i == len(self.process_all):
                model['Sink'] = Sink(env)
            else:
                model[self.process_all[i]] = Process(env, self.process_all[i], self.machine_num[self.process_all[i]],
                                                     self.priority[self.process_all[i]],
                                                     model, self.process_list)

        return env, model, source

    def step(self, action):
        self.done = False

        if len(self.model['BH'].buffer_to_machine.items) != 0 and len(
                self.model['BH'].machine_store.items) != 0:
            self.model['BH'].action = action
            dispatching_possible = True
        else:
            self.model['BH'].action = action
            dispatching_possible = False
            #     self.model[process].action = 4 # action of doing nothing

        current_time_step = self.env.now

        parts_in_buffer = self.model['BH'].buffer_to_machine.items[:]
        parts_in_machine = []
        for i, machine in enumerate(self.model['BH'].machines):
            if len(machine.part_in_machine) != 0:
                parts_in_machine.append(machine.part_in_machine[0])

        current_new_idles = self.model['BH'].new_idles[:]
        current_new_arrivals = self.model['BH'].new_arrivals[:]

        while True:
            # Go to next time step
            # any new arrival event or new idle machine event
            new_conditions = (len(current_new_arrivals) != len(self.model['BH'].new_arrivals)
                                  or len(current_new_idles) != len(self.model['BH'].new_idles))
            new_time_step_possible = new_conditions  # any and all
            if new_time_step_possible:
                break

            # 중간에 시뮬레이션이 종료되는 경우 break
            if self.model['BH'].parts_routed == self.part_num:
                self.done = True
                self.env.run()
                break

            self.env.step()

        if self.model['BH'].parts_routed == self.part_num:
            self.done = True

        if self.done:
            self.env.run()

        if len(self.model['BH'].buffer_to_machine.items) != 0 and len(
                self.model['BH'].machine_store.items) != 0:
            next_dispatching_possible = True
        else:
            next_dispatching_possible = False

        next_time_step = self.env.now

        next_state = self._get_state()
        reward = self._calculate_reward('BH', current_time_step, next_time_step, dispatching_possible, action,
                                            parts_in_machine,
                                            parts_in_buffer)

        return next_state, reward, self.done, next_dispatching_possible


    def _get_state(self):
        # f_1 (feature 1)
        # f_2 (feature 2)
        # f_3 (feature 3)
        # f_4 (feature 4)
        # aid = np.array([i])
        f_1 = np.zeros(len(self.job_type))
        NJ = np.zeros(len(self.job_type))

        f_2 = np.zeros(len(self.model['BH'].machines))

        z = np.zeros(len(self.model['BH'].machines))
        f_3 = np.zeros(len(self.model['BH'].machines))

        f_4 = np.zeros(len(self.model['BH'].machines))

        machine_list = []
        # for process in self.process_all:
        for i, machine in enumerate(self.model['BH'].machines):
            machine_list.append(machine)
        # for i, machine in enumerate(self.model[process].machines):
        #     machine_list.append(machine)

        for i, machine in enumerate(machine_list):
            if len(machine.part_in_machine) != 0:  # if the machine is not idle(working)
                # print(machine.machine.items)
                step = machine.part_in_machine[0].step

                # feature 2
                f_2[i] = machine.part_in_machine[0].type / len(self.job_type)

                # feature 4
                f_4[i] = (machine.part_in_machine[0].due_date['BH'] - self.env.now) / \
                         machine.part_in_machine[0].process_time[
                             machine.process_name]
                # z_i : remaining process time of part in machine i
                if machine.part_in_machine[0].real_proc_time > self.env.now - machine.start_work:
                    z[i] = machine.part_in_machine[0].real_proc_time - (self.env.now - machine.start_work)
                    # feature 3
                    f_3[i] = z[i] / machine.part_in_machine[0].process_time[machine.process_name]

        # features to represent the tightness of due date allowance of the waiting jobs
        # f_5 (feature 5)
        # f_6 (feature 6)
        # f_7 (feature 7)
        # f_8 (feature 8)
        # f_9 (feature 9)
        f = [[] for _ in range(len(self.job_type))]
        f_5 = np.zeros(len(self.job_type))
        f_6 = np.zeros(len(self.job_type))
        f_7 = np.zeros(len(self.job_type))

        f_8 = np.zeros((len(self.job_type), 4))
        f_9 = np.zeros(len(self.job_type))
        # f_9 = np.zeros((len(self.job_type), 4))

        if len(self.model['BH'].buffer_to_machine.items) == 0:
            f_5 = np.zeros(len(self.job_type))
            f_6 = np.zeros(len(self.job_type))
            f_7 = np.zeros(len(self.job_type))

            f_8 = f_8.flatten()
            # f_9 = np.zeros(len(self.job_type)*4)

        else:
            # interval number indicating the tightness of the due date allowance
            g = np.zeros((len(self.job_type), 4))
            # interval number indicating the process time
            h = np.zeros((len(self.job_type), 4))

            for i, part in enumerate(self.model['BH'].buffer_to_machine.items):

                NJ[part.type] += 1
                f[part.type].append(
                    (part.due_date['BH'] - self.env.now) / part.process_time[part.process_list[part.step]])
                # print('hi ', part.due_date - self.env.now)
                # print(part.due_date)
                # print(self.env.now)
                # print('ho ',part.process_time)

                # case for interval number g
                if (part.due_date['BH'] - self.env.now) >= part.max_process_time[part.process_list[part.step]]:
                    g[part.type][0] += 1
                elif (part.due_date['BH'] - self.env.now) >= part.min_process_time[part.process_list[part.step]] \
                        and (part.due_date['BH'] - self.env.now) < part.max_process_time[part.process_list[part.step]]:
                    g[part.type][1] += 1
                elif (part.due_date['BH'] - self.env.now) >= 0 and (part.due_date['BH'] - self.env.now) < \
                        part.min_process_time[
                            part.process_list[part.step]]:
                    g[part.type][2] += 1
                elif (part.due_date['BH'] - self.env.now) < 0:
                    g[part.type][3] += 1

            # feature 1
            f_1 = np.array([2 ** (-1 / nj) if nj > 0 else 0 for nj in NJ])

            # feature 8
            for j in self.job_type:
                for _g in range(4):
                    f_8[j][_g] = 2 ** (-1 / g[j][_g]) if g[j][_g] != 0 else 0
            f_8 = f_8.flatten()

            for i in range(len(self.job_type)):
                if len(f[i]) != 0:
                    min_tightness = np.min(np.array(f[i]))
                    max_tightness = np.max(np.array(f[i]))
                    avg_tightness = np.average(np.array(f[i]))
                else:
                    min_tightness = 0
                    max_tightness = 0
                    avg_tightness = 0

                # feature 5
                f_5[i] = min_tightness
                # feature 6
                f_6[i] = max_tightness
                # feature 7
                f_7[i] = avg_tightness

        # Calculating mean-weighted-tardiness
        # Calculating make span
        if len(self.model['Sink'].sink) != 0:
            mean_w_tardiness = 0
            make_span = self.model['Sink'].sink[0].completion_time['BH']
            for part in self.model['Sink'].sink:
                filter_list = list(filter(lambda x: x.type == part.type, self.model['Sink'].sink))
                f_9[part.type] += part.weight * min(0, part.due_date['BH'] - part.completion_time['BH']) / len(
                    filter_list)
                mean_w_tardiness += part.weight * max(0, part.completion_time['BH'] - part.due_date['BH'])

            self.mean_weighted_tardiness = mean_w_tardiness / len(self.model['Sink'].sink)
        f_9 = f_9 / 100

        state = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8), axis=None)

        # Calculating mean-weighted-tardiness
        # Calculating make span
        if len(self.model['Sink'].sink) != 0:
            mean_w_tardiness = 0
            make_span = self.model['Sink'].sink[0].completion_time['BH']
            for part in self.model['Sink'].sink:
                mean_w_tardiness += part.weight * max(0, part.completion_time['BH'] - part.due_date['BH'])
                if part.completion_time['BH'] > make_span:
                    make_span = part.completion_time['BH']

            self.mean_weighted_tardiness = mean_w_tardiness / len(self.model['Sink'].sink)
            self.make_span = make_span

        return state


    def reset(self):
        self.env, self.model, self.source = self._modeling()
        self.done = False

        current_new_idles = self.model['BH'].new_idles[:]
        current_new_arrivals = self.model['BH'].new_arrivals[:]
        while True:
            # Go to next time step
            # any new arrival event or new idle machine event
            new_conditions = (len(current_new_arrivals) != len(self.model['BH'].new_arrivals)
                              or len(current_new_idles) != len(self.model['BH'].new_idles))
            new_time_step_possible = new_conditions  # any and all
            if new_time_step_possible:
                break

            # 중간에 시뮬레이션이 종료되는 경우 break
            if self.model['BH'].parts_routed == self.part_num:
                self.done = True
                self.env.run()
                break

            self.env.step()

        if len(self.model['BH'].buffer_to_machine.items) != 0 and len(
                self.model['BH'].machine_store.items) != 0:
            next_dispatching_possible = True
        else:
            next_dispatching_possible = False

        return self._get_state(), next_dispatching_possible

    def _calculate_reward(self, process, current_time_step, next_time_step, dispatching_possible, action,
                          parts_in_machine,
                          parts_in_buffer):
        # calculate reward for parts in waiting queue
        sum_reward_for_tardiness = 0
        sum_reward_for_makespan = 0
        total_weight_sum = 0
        for part in parts_in_buffer:
            if part.completion_time[process] == None:
                if part.due_date[process] < current_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - current_time_step)
                elif part.due_date[process] >= current_time_step and part.due_date[process] < next_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - part.due_date[process])
            else:
                if part.due_date[process] < current_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (part.completion_time[process] - current_time_step)
                elif part.due_date[process] >= current_time_step and part.due_date[process] < next_time_step:
                    sum_reward_for_tardiness += part.weight * (part.due_date[process] - part.completion_time[process])

        for part in parts_in_machine:
            if part.completion_time[process] == None:
                if part.due_date[process] < current_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - current_time_step)
                elif part.due_date[process] >= current_time_step and part.due_date[process] < next_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - part.due_date[process])
            else:
                if part.due_date[process] < current_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (part.completion_time[process] - current_time_step)
                elif part.due_date[process] >= current_time_step and part.due_date[process] < next_time_step:
                    sum_reward_for_tardiness += part.weight * (part.due_date[process] - part.completion_time[process])

        if len(self.model['Sink'].sink) != 0:
            mean_w_tardiness = 0
            make_span = self.model['Sink'].sink[0].completion_time['BH']
            for part in self.model['Sink'].sink:
                # mean_w_tardiness += part.weight * max(0, part.completion_time - part.due_date)
                # mean_w_tardiness += part.weight * min(0, part.due_date - part.completion_time)
                if part.completion_time['BH'] > make_span:
                    make_span = part.completion_time['BH']

            # sum_reward_for_tardiness = mean_w_tardiness / len(self.model['Sink'].sink)
            # sum_reward_for_tardiness = sum_reward_for_tardiness / 100
            sum_reward_for_makespan = make_span

        sum_reward = sum_reward_for_makespan + sum_reward_for_tardiness

        return sum_reward_for_tardiness


if __name__ == '__main__':


    forming_shop = Forming()

    #print(len(forming_shop.model['LH'].buffer_to_machine.items))
    #print(len(forming_shop.model['LH'].machine_store.items))
    for i in range(500):
        next_state1, reward1, done1, next_dispatching = forming_shop.step(0)

        print(len(forming_shop.model['BH'].buffer_to_machine.items))
        print(len(forming_shop.model['BH'].machine_store.items))
        print(len(forming_shop.model['Sink'].sink))
        print(next_state1)
        print(reward1)
        print(done1)

    print(forming_shop.mean_weighted_tardiness)

    #print(forming_shop.model['Sink'].sink[0].completion_time)
    # LH가 직전 공정인 BH보다 capacity가 크기 때문에 LH가 병목공정이 되고
    # buffer to machine 에는 part가 많이 쌓이는 반면 idle machine은 하나씩 발생하여
    # idle machine 하나가 발생하면 어떤 part를 먼저 해당 idle machine에 투입할 지 agent가 결정해준다.
