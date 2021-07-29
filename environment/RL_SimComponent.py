import simpy
import os
import random
import numpy as np
import math
from collections import OrderedDict
import functools


class Part(object):
    def __init__(self, name, type, process_time, weight, K, process_list):
        # 해당 Part 번호
        self.id = name

        # 해당 Part type
        self.type = type

        # average process time
        # p_ij = {'BH': np.random.uniform(1, 20, size=(len(job_type), machine_num['BH'])),
        # 'LH': np.random.uniform(1, 20, size=(len(job_type), machine_num['LH']))} ->  이 형태로 process time 입력
        # process_time => average process time
        self.process_time = {'BH': np.average(process_time['BH'], axis=1)[self.type]}
        # determined process time
        self.real_proc_time = None

        # list of process that this part go through
        self.process_list = process_list

        # machine 별 avg process time
        self.p_ij = {'BH': process_time['BH'][self.type][:]}
        self.max_process_time = {'BH': np.max(np.array(self.p_ij['BH']))}
        self.min_process_time = {'BH': np.min(np.array(self.p_ij['BH']))}
        self.total_process_time = self.process_time['BH']

        # 작업을 완료한 공정의 수
        self.step = 0

        # part의 납기일 정보
        self.due_date = {}
        # due date generating factor
        self.K = K

        self.W_ij = None

        self.weight = weight

        self.completion_time = {'BH': None}

    def set_due_date(self, arrival_time):
        for process in self.process_list[:-1]:
            self.due_date[process] = arrival_time + self.K[process] * self.process_time[process]
            arrival_time += self.process_time[process]


class Source(object):
    def __init__(self, env, IAT, weight, job_type, job_type_list, process_time, model, process_list, machine_num, part_num, K):
        self.env = env
        self.name = 'Source'

        self.parts_sent = 0
        self.IAT = IAT

        # job type 별 특징
        self.weight = weight
        self.job_type = job_type
        self.job_type_list = job_type_list
        # avg process time
        # process_time = {'BH':[1.8, 2.4, 3.2, 3.5, 4.5, 5, 5.3, 6, 6.3, 6.5], 'LH':[10, 12, 15, 17, 19, 22, 24, 27, 30, 32]} 형태
        self.process_time = process_time

        # for idx in range(10):
        env.process(self.job_generating_process(self.IAT, self.job_type_list))
        # env.process(self.run())

        # 각 job type 별 생성된 job 수
        self.generated_job_types = np.zeros(len(self.job_type))

        self.model = model
        self.process_list = process_list

        self.machine_num = machine_num
        self.part_num = part_num
        self.K = K

    def job_generating_process(self, IAT, job_type_list):
        while True:
            for jb_type, iat in zip(job_type_list, IAT):
                yield self.env.timeout(iat)
                self.generated_job_types[jb_type] += 1
                w = self.weight[jb_type]
                p = self.process_time
                # generate job
                part = Part(name='job{0}_{1}'.format(jb_type, self.generated_job_types[jb_type]), type=jb_type,
                            process_time=p, weight=w, K=self.K, process_list=self.process_list)
                part.set_due_date(self.env.now)

                # put job batch to next process buffer_to_machine
                self.model[self.process_list[part.step]].buffer_to_machine.put(part)

                self.model[self.process_list[part.step]].new_arrivals.append(part)

                # print(self.env.now, '   Source   ', len(self.model[self.process_list[0]].buffer_to_machine.items))
                self.parts_sent += 1

            if self.parts_sent == self.part_num:
                break


########################################################################################################################

class Process(object):
    def __init__(self, env, name, machine_num, priority, model, process_list, capacity=float('inf'),
                 capa_to_machine=float('inf'), capa_to_process=float('inf')):
        self.env = env
        self.name = name
        self.model = model
        self.machine_num = machine_num

        self.process_list = process_list

        self.capa = capacity
        self.priority = priority
        self.routing_ongoing = False

        self.parts_sent = 0
        self.parts_routed = 0
        self.new_arrivals = []
        self.new_idles = []

        self.buffer_to_machine = simpy.FilterStore(env, capacity=capa_to_machine)
        self.buffer_to_process = simpy.Store(env, capacity=capa_to_process)

        self.machine_store = simpy.FilterStore(env, capacity=machine_num)

        self.machines = [Machine(env, self.name, 'Machine_{0}'.format(i), idx=i, priority=self.priority[i],
                                 out=self.buffer_to_process, model=model, process_list=self.process_list) for i in range(machine_num)]

        for i in range(self.machine_store.capacity):
            self.machine_store.put(self.machines[i])

        env.process(self._to_machine())
        env.process(self.check_idle_machine())

        # idle machine 이 있을 때 열리고(succeed) 없을 때 닫혀있는 스위치(valve?) event
        # check_idle_machine(Process)에 의해서 제어 됨
        self.idle_machine = env.event()
        # idle machine 을 check할 시점이 되면 열리고(succeed) 없을 때 닫혀있는 스위치(valve?) event
        self.wait_before_check = env.event()

        self.action = 0  # env.step에서는 Process0의 self.action을 매 step마다 선택

        self.working_process_list = dict()  # 현재 작업 진행중인 Process list

        # self.parts_arriving_event = False
        # self.idle_machines_event = False
        self.parts_arriving_event = self.env.event()
        self.idle_machines_event = [self.env.event() for _ in range(machine_num)]

    def _to_machine(self):
        self.i = 0
        step = 0
        while True:
            # print("++++++++++++++++++++ " + self.name + '  {0}th iteration'.format(self.i) + " ++++++++++++++++++++")
            # wait until there exists an idle machine
            self.routing_logic = None
            self.i += 1
            # yield self.idle_machine     # idle machine 있을 때만 열리는 valve
            # print('Hi')

            # If there exist idle machines and also parts in buffer_to_machine
            # Then take action (until one of the items becomes zero)
            if len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                while len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                    self.routing_logic = self.routing(self.action, self.name)
                    self.routing_ongoing = True

                    # print(self.env.now, '   HEEEE', self.name, '  case 1')

                    idle = [x.name for x in self.machine_store.items]
                    # print(str(self.env.now) + '  ' + self.name + ' case 1 idle machines : ')
                    # print(idle)
                    # print(str(self.env.now) + ' ' + self.name + '  case 1 parts in buffer : ')
                    # print([x.id for x in self.buffer_to_machine.items])

                    part = yield self.buffer_to_machine.get()
                    # print(part)

                    machine = yield self.machine_store.get()
                    # print('HEEEE', self.name, '  case 1')
                    ################################ Processing Process generated ##########################
                    self.parts_routed += 1
                    machine.part_in_machine.append(part)
                    self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))

                # print(self.env.now, '  ', self.name, '  ', 'step{0} Routing finished'.format(step))
                step += 1
                self.routing_ongoing = False

            ############### case 2 가 문제임 #########################
            elif len(self.buffer_to_machine.items) == 0 and len(self.machine_store.items) != 0:
                # print(self.env.now, '  hello  ', self.name)
                # print(self.env.now, 'iiiiiiiiii   ', self.name, '   ', len(self.buffer_to_machine.items))
                part = yield self.buffer_to_machine.get()
                # print(self.env.now, "  I got it  ", self.name)
                self.buffer_to_machine.put(part)
                self.routing_logic = self.routing(self.action, self.name)
                self.routing_ongoing = True
                part = yield self.buffer_to_machine.get()
                machine = yield self.machine_store.get()

                self.parts_routed += 1
                machine.part_in_machine.append(part)
                self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))


                # self.buffer_to_machine.put(part)
                # while len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                #     self.routing_logic = self.routing(self.action)
                #     self.routing_ongoing = True
                #
                #     idle = [x.name for x in self.machine_store.items]
                #     print(str(self.env.now) + '  ' + self.name + ' case 2  idle machines : ')
                #     print(idle)
                #     print(str(self.env.now) + ' ' + self.name + ' case 2  parts in buffer : ')
                #     print([x.id for x in self.buffer_to_machine.items])
                #
                #     part = yield self.buffer_to_machine.get()
                #     machine = yield self.machine_store.get()
                #
                #     self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))
                #
                #     self.parts_routed += 1
                self.routing_ongoing = False
            ####### case 3 도 문제임 ####################################
            else:
                idle = [x.name for x in self.machine_store.items]
                # print(str(self.env.now) + '  ' + self.name + ' case 3  idle machines : ')
                # print(idle)
                # print(str(self.env.now) + ' ' + self.name + ' case 3  parts in buffer : ')
                # print([x.id for x in self.buffer_to_machine.items])
                # print('HAAAAAAAAAAAAAAAAA', self.name, '  case 3')
                yield self.idle_machine     # idle machine 있을 때만 열리는 valve
                # print("sdfsgerheriopkporkgrgrg")



    def to_machine(self):
        if len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
            while len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                self.routing_logic = self.routing(self.action, self.name)

                idle = [x.name for x in self.machine_store.items]

                part = yield self.buffer_to_machine.get()

                machine = yield self.machine_store.get()

                self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))

                self.parts_routed += 1



    def check_idle_machine(self):  # idle_machine event(valve)를 제어하는 Process
        while True:
            if len(self.machine_store.items) != 0:
                # print('machine idle  ', self.name)
                # idle machine 있을 시 idle_machine event(valve)를 열었다가 바로 닫는다.
                self.idle_machine.succeed()
                self.idle_machine = self.env.event()
            # print(self.env.now, '  Wait for checking')
            # print(len(self.machine_store.items), self.name)
            # print(self.env.now)
            # print(len(self.buffer_to_machine.items), self.name)
            yield self.wait_before_check  # wait for time to check(valve)
            # print('yeeeeeeeeeeeeeeeeeeee')
            # machine 이 반납된 후마다 열렸다가 바로 닫힘

    # i : idle machine index , j : part in buffer_to_machine index
    # idle_machines (list) 와 parts_in_buffer (list) 이용해서 indexing
    # idle machine name 과 part in buffer_to_machine id
    def routing(self, action, process):
        if action == 0:  # WSPT
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = p_ij / j.weight
                        W_i.append(w_ij)
                    i.W_ij = min(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = p_ij / j.weight
                        W_j.append(w_ij)
                    j.W_ij = min(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij)

            elif len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) == 0:
                self.machine_store.items.sort(key=lambda machine: machine.priority)


        elif action == 1:  # WMDD
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = max(p_ij, j.due_date[self.process_list[j.step]] - self.env.now)
                        w_ij = w_ij / j.weight
                        W_i.append(w_ij)
                    i.W_ij = min(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = max(p_ij, j.due_date[self.process_list[j.step]] - self.env.now)
                        w_ij = w_ij / j.weight
                        W_j.append(w_ij)
                    j.W_ij = min(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij)

            elif len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) == 0:
                self.machine_store.items.sort(key=lambda machine: machine.priority)

            return 1

        elif action == 2:  # ATC
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                p = 0.0  # average nominal processing time
                for part in self.buffer_to_machine.items:
                    p += part.process_time[self.process_list[part.step]]
                p = p / len(self.buffer_to_machine.items)

                h = 2.3  # look-ahead parameter

                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = -1 * max(0, j.due_date[self.process_list[j.step]] - self.env.now - p_ij) / (h * p)
                        w_ij = j.weight / p_ij * math.exp(w_ij)
                        W_i.append(w_ij)
                    i.W_ij = max(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij, reverse=True)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = -1 * max(0, j.due_date[self.process_list[j.step]] - self.env.now - p_ij) / (h * p)
                        w_ij = j.weight / p_ij * math.exp(w_ij)
                        W_j.append(w_ij)
                    j.W_ij = max(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij, reverse=True)

            elif len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) == 0:
                self.machine_store.items.sort(key=lambda machine: machine.priority)


        elif action == 3:  # WCOVERT
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                K_t = 2.3  # approximation factor

                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = 1 - max(0, j.due_date[self.process_list[j.step]] - self.env.now - p_ij) / (K_t * p_ij)
                        w_ij = j.weight / p_ij * max(0, w_ij)
                        W_i.append(w_ij)
                    i.W_ij = max(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij, reverse=True)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = 1 - max(0, j.due_date[self.process_list[j.step]] - self.env.now - p_ij) / (K_t * p_ij)
                        w_ij = j.weight / p_ij * max(0, w_ij)
                        W_j.append(w_ij)
                    j.W_ij = max(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij, reverse=True)

            elif len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) == 0:
                self.machine_store.items.sort(key=lambda machine: machine.priority)

        elif action == 4: # do nothing when there are no new part arrivals or no idle machines
            pass # do nothing

class Machine(object):
    def __init__(self, env, process_name, name, idx, priority, out, model, process_list):
        self.env = env
        self.process_name = process_name
        self.name = name
        self.idx = idx
        self.priority = priority
        self.out = out
        self.model = model
        self.process_list = process_list

        # self.machine = simpy.Store(env, capacity=1)
        # self.machine = simpy.Resource(env, capacity=1)
        self.part_in_machine = []

        self.start_work = None
        self.working_start = 0.0

        self.W_ij = None

    def work(self, part, machine_store):
        # process_time
        proc_time = part.p_ij[self.process_name][self.idx]
        # proc_time = np.random.triangular(left=0.9 * proc_time, mode=proc_time, right=1.1 * proc_time)
        part.real_proc_time = proc_time

        self.start_work = self.env.now
        self.working_start = self.env.now
        # self.part_in_machine.append(part)

        yield self.env.timeout(proc_time)

        _ = self.part_in_machine.pop(0)
        # return machine object(self) to the machine store
        machine_store.put(self)

        # next process
        next_process_name = self.process_list[part.step + 1]
        # print(str(self.env.now) + '  ' + part.id + '  next process is : '+ next_process_name)
        next_process = self.model[next_process_name]
        process = self.model[self.process_name]

        if next_process.__class__.__name__ == 'Process':
            # part transfer
            part.completion_time[self.process_name] = self.env.now
            next_process.buffer_to_machine.put(part)
            next_process.new_arrivals.append(part)
            process.new_idles.append(part)
        else:
            part.completion_time[self.process_name] = self.env.now
            next_process.put(part)
            process.new_idles.append(part)

        part.step += 1
        self.model[self.process_name].parts_sent += 1

        # Idle machine occurs -> check
        # Idle machine check 시점이 되면 wait_before_check(valve)를 열었다가 바로 닫는다
        self.model[self.process_name].wait_before_check.succeed()
        self.model[self.process_name].wait_before_check = self.env.event()

        # time step when idle machine event occurs
        self.model[self.process_name].idle_machines_event[self.idx].succeed()
        self.model[self.process_name].idle_machines_event[self.idx] = self.env.event()


class Sink(object):
    def __init__(self, env):
        self.env = env
        self.name = 'Sink'

        # self.tp_store = simpy.FilterStore(env)  # transporter가 입고 - 출고 될 store
        self.parts_rec = 0
        self.last_arrival = 0.0
        self.sink = list()

    def put(self, part):
        self.parts_rec += 1
        self.last_arrival = self.env.now
        self.sink.append(part)
        # print(str(self.env.now) + '  ' + self.name + '  ' + part.id + '  completed')



