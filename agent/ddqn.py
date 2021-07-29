import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from collections import deque
from environment.Parallel_Machine import Forming



class Network(tf.keras.Model):
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


class DDQN():
    def __init__(self, state_size, action_size, model_path=None, load_model=False):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 5000
        self.target_update_iter = 20

        self.memory = deque(maxlen=10000)

        if load_model:
            self.model = Network(action_size)
            self.target_model = Network(action_size)
            self.model.load_weights(model_path)
            self.target_model.load_weights(model_path)
        else:
            self.model = Network(action_size)
            self.target_model = Network(action_size)
            self.update_target_model()
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, clipvalue=1.0)

        self.avg_q_max, self.avg_loss = 0, 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def append_sample(self, obs, action, reward, next_obs, done, next_dispatching):
        self.memory.append((obs, action, reward, next_obs, done, next_dispatching))

    def get_action(self, state, dispatching):
        state = np.reshape(state, [1, -1])
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            if dispatching:
                q_value = self.model(state)
                action = np.argmax(q_value[0][:-1])
            else:
                action = self.action_size - 1

        self.action = action
        return action


    def train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])
        next_dispatchings = np.array([sample[5] for sample in mini_batch])

        model_params = self.model.trainable_variables
        best_actions = []
        with tf.GradientTape() as tape:
            for i, (state, action, reward, next_state, done, next_dispatching) in enumerate(
                    zip(states, actions, rewards, next_states,
                        dones, next_dispatchings)):
                state = np.reshape(state, [1, -1])
                next_state = np.reshape(next_state, [1, -1])
                next_predict = self.model(next_state)
                if next_dispatching:
                    best_action = np.argmax(next_predict[0][:-1])
                else:
                    best_action = self.action_size - 1
                best_actions.append(best_action)
                # best_actions = tf.stop_gradient(best_actions)
                # action = np.reshape(action, [1, -1])
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)
            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)
            predicts = tf.convert_to_tensor(predicts)

            targets = rewards + (1 - dones) * self.discount_factor \
                      * np.array([target_predicts[i, best_actions[i]] for i in range(target_predicts.shape[0])])
            loss = tf.reduce_mean(tf.square(targets - predicts))

            self.avg_loss += loss.numpy()

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
    num_episode = 50000

    load_model = True

    score_avg = 0

    state_size = 95
    action_size = 5

    model_path = '../model/ddqn/queue-%d' % action_size
    summary_path = '../summary/ddqn/queue-%d' % action_size
    result_path = '../result/ddqn/queue-%d' % action_size
    event_path = '../simulation/ddqn/queue-%d' % action_size

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(event_path):
        os.makedirs(event_path)

    env = Forming()

    agent = DDQN(state_size, action_size)
    writer = tf.summary.create_file_writer(summary_path)

    avg_max_q_list = []
    reward_list = []
    make_span_list = []
    mean_weighted_tardiness_list = []
    loss_list = []
    for e in range(1, num_episode):
        done = False
        step = 0
        episode_reward = 0
        state, dispatching_possible = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            step += 1

            action = agent.get_action(state, dispatching_possible)
            next_state, reward, done, next_dispatching_possible = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.avg_q_max += np.amax(agent.model(state)[0])

            episode_reward += reward

            agent.append_sample(state, action, reward, next_state, done, next_dispatching_possible)

            if len(agent.memory) >= agent.train_start:
                agent.train()

            if e % agent.target_update_iter == 0:
                agent.update_target_model()

            state = next_state
            dispatching_possible = next_dispatching_possible

            if done:

                mean_weighted_tardiness = env.mean_weighted_tardiness
                make_span = env.make_span

                score_avg = 0.9 * score_avg + 0.1 * episode_reward if score_avg != 0 else episode_reward

                with writer.as_default():
                    tf.summary.scalar('Loss/Average Loss', agent.avg_loss / float(step), step=e)
                    tf.summary.scalar('Performance/Average Max Q', agent.avg_q_max / float(step), step=e)
                    tf.summary.scalar('Performance/Reward', episode_reward, step=e)
                    tf.summary.scalar("Perf/Mean weighted tardiness", mean_weighted_tardiness, step=e)
                    tf.summary.scalar("Perf/Make span", make_span, step=e)
                    avg_max_q_list.append(agent.avg_loss / float(step))
                    reward_list.append(episode_reward)
                    make_span_list.append(make_span)
                    mean_weighted_tardiness_list.append(mean_weighted_tardiness)
                    loss_list.append(agent.avg_loss / float(step))

                if e % 250 == 0:
                    agent.model.save_weights(model_path, save_format='tf')
                    print("Saved Model at episode %d" % e)

                agent.avg_q_max, agent.avg_loss = 0, 0

                print("episode: {:4d} | score_avg: {:5.4f} | memory_length: {:4d} | epsilon: {:.4f}".format(
                    e, score_avg, len(agent.memory), agent.epsilon
                ))

    log_data = pd.DataFrame({"avg_max_q_": avg_max_q_list, "reward": reward_list, "lead_time": make_span_list,
                             "mean_weighted_tardiness": mean_weighted_tardiness_list, "loss": loss_list})
    log_data.to_csv(summary_path + "/data.csv")
