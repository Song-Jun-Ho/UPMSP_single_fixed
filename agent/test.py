import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from collections import deque
from environment.Parallel_Machine import Forming

class DDQN(tf.keras.Model):
    def __init__(self, a_size):
        super().__init__(name='ddqn')
        self.hidden1 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")
        self.hidden2 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")
        self.hidden3 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.out = tf.keras.layers.Dense(a_size)

    def call(self, inputs):
        hidden1 = self.hidden1(inputs)
        hidden2 = self.hidden2(hidden1)
        hidden3 = self.hidden3(hidden2)
        q_values = self.out(hidden3)
        return q_values


class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model_path = '../model/ddqn/queue-%d' % action_size

        self.model = DDQN(self.action_size)
        self.model.load_weights(self.model_path)

    def get_action(self, state):
            q_value = self.model(state)
            return np.argmax(q_value[0])


if __name__ == "__main__":

    state_size = 104
    action_size = 4

    # agent = DDQNAgent(state_size=state_size, action_size=action_size)

    job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weight = np.random.uniform(0, 5, len(job_type))
    machine_num = {'LH': 8}
    # job type 별 average process time
    # process_time = [5, 9, 12, 16, 18, 23, 25, 28, 30, 32]
    p_ij = np.random.uniform(1, 20, size=(len(job_type), machine_num['LH']))
    p_j = np.average(p_ij, axis=1)
    process_list = ['LH', 'Sink']
    process_all = ['LH']
    priority = {'LH': [1, 2, 3, 4, 5, 6, 7, 8]}
    # IAT = 5
    arrival_rate = machine_num['LH'] / np.average(p_j)
    IAT = 1 / arrival_rate
    part_num = 1000
    # due date generating factor
    K = 1

    env = Forming(weight=weight, job_type=job_type, machine_num=machine_num, process_time=p_ij,
                  process_list=process_list, process_all=process_all, priority=priority, IAT=IAT, part_num=part_num, K=K)

    num_episode = 100
    mean_weighted_tardiness_list = []
    for e in range(num_episode):
        np.random.seed(e)
        done = False
        epsisode_reward = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        action_list = []

        while not done:
            # action = agent.get_action(state)
            # action_list.append(action)
            #
            # next_state, reward, done = env.step(action)
            next_state, reward, done = env.step(1)

            epsisode_reward += reward

            state = next_state
            state = np.reshape(state, [1, state_size])

            if done:
                mean_weighted_tardiness = env.mean_weighted_tardiness
                mean_weighted_tardiness_list.append(mean_weighted_tardiness)
                print("episode: {:3d} | episode reward: {:5.4f} | mean weighted tardiness: {:5.4f}".format(e, epsisode_reward, mean_weighted_tardiness))
                # print(action_list)

    mean_weighted_tardiness_list = np.array(mean_weighted_tardiness_list)
    avg_mean_weighted_tardiness = np.average(mean_weighted_tardiness_list)
    std_mean_weighred_tardiness = np.std(mean_weighted_tardiness_list)
    print('Average mean weighted tardiness : ', avg_mean_weighted_tardiness)
    print('Std mean weighted tardiness : ', std_mean_weighred_tardiness)
    print(mean_weighted_tardiness_list)

    # RL Average mean weighted tardiness :  31.20153411297188
    # SPT Average mean weighted tardiness :  31.3427262183924
    # WMDD Average mean weighted tardiness :  36.390683514505
    # ATC Average mean weighted tardiness :  41.66234072589088
    # WCOVERT Average mean weighted tardiness :  51.14990421942022

