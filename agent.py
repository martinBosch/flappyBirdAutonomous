import os.path
import random
from collections import deque

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential, Input
from keras.optimizer_v2.adam import Adam

from flappybird import FlappyBird

DO_NOTHING = 0
FLAP = 1


def take_metrics(scores, steps_until_dead, pipes):
    if len(scores) % 100 == 0:
        pd.DataFrame({'Scores': scores}).plot().get_figure().savefig('scores.pdf')
        pd.DataFrame({'Loss': steps_until_dead}).plot().get_figure().savefig('steps_until_dead.pdf')
        pd.DataFrame({'Pipes': pipes}).plot().get_figure().savefig('pipes_until_dead.pdf')


def must_update_target_network_weights(n):
    return n % 1000 == 0


def do_something_under(q_value_flap, q_value_do_nothing):
    if q_value_flap > q_value_do_nothing:
        return FLAP
    else:
        return DO_NOTHING


def normalize(value, scale):
    return value / scale


class Agent:
    def __init__(self):
        self.flappybird = FlappyBird(self)

        self.alpha = 0.7
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.01
        self.learning_rate = 0.01
        self.pipes = 0

        self.q_network = DeepQNetwork(weights_file="q_network_weights.h5", learning_rate=self.learning_rate)
        self.target_network = DeepQNetwork(weights_file="target_network_weights.h5", learning_rate=self.learning_rate)
        self.memory = deque(maxlen=4000)

        self.target_network.update_weights(self.q_network)

    def init(self):
        pass

    def save(self):
        self.q_network.save_weights()
        self.target_network.save_weights()

    """
     * Method used to determine the next move to be performed by the agent.
     * now is moving random
     """

    def execute_action_by(self, state):
        action = self.get_action_by(state)
        if action == FLAP:
            self.flappybird.hold_key_down()
        else:
            self.flappybird.release_key()
        return action

    def get_action_by(self, state):
        if np.random.uniform() < self.epsilon + self.epsilon_min:
            return do_something_under(np.random.uniform(), np.random.uniform())
        else:
            q_value = self.q_network.rewards_for(state)
            return do_something_under(q_value[FLAP], q_value[DO_NOTHING])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = self.take_a_sample_of(sample_batch_size)

        x = []
        y = []

        for state, action, reward, next_state, done in sample_batch:
            immediate_reward = reward
            future_reward = self.gamma * max(self.target_network.rewards_for(next_state))
            rewards_prediction = self.q_network.rewards_for(state)
            rewards_prediction[action] = (1 - self.alpha) * rewards_prediction[action] + self.alpha * (
                    immediate_reward + future_reward)
            x.append(state)
            y.append(rewards_prediction)
        self.q_network.save_rewards_for(x, y)

    def take_a_sample_of(self, sample_batch_size):
        return random.sample(self.memory, sample_batch_size)

    def state(self):
        positions = self.flappybird.get_world_position_objets()
        bottom_pipe = positions[0]
        bird = positions[2]

        x = bottom_pipe[0] - (bird[0] + bird[2])
        y_bottom = bottom_pipe[1] - bird[1]

        return np.array([normalize(x, self.flappybird.screen_width), normalize(y_bottom, self.flappybird.screen_high)])

    def reward(self):
        state = self.state()
        x = state[0]

        if 0.0030 >= x > 0.0000:
            self.pipes += 1

        if not self.flappybird.dead and x == 0:
            return 10.0
        if not self.flappybird.dead:
            return 0.1 + 1 * self.pipes
        else:
            allowance = 1 * self.pipes
            if allowance > 100:
                return -100
            return -100.0 + allowance

    def run(self):
        self.flappybird.init_game()
        batch_size = 64

        n = 0
        score = 0
        scores = []
        step_until_dead = 0
        steps_until_dead = []
        pipes_until_dead = []

        while True:
            state = self.state()
            action = self.execute_action_by(state)

            self.flappybird.each_cycle()

            new_state = self.state()
            immediate_reward = self.reward()
            score += immediate_reward

            self.remember(
                state=state,
                action=action,
                reward=immediate_reward,
                next_state=new_state,
                done=self.flappybird.dead
            )

            self.decrease_epsilon()

            if n % 10 == 0:
                self.replay(sample_batch_size=batch_size)

            if must_update_target_network_weights(n):
                self.target_network.update_weights(self.q_network)

            if self.flappybird.dead:
                self.flappybird.restart_game()
                scores.append(score)
                steps_until_dead.append(step_until_dead)
                pipes_until_dead.append(self.pipes)
                take_metrics(scores, steps_until_dead, pipes_until_dead)
                print(score)
                score = 0
                step_until_dead = 0
                self.pipes = 0
            else:
                step_until_dead += 1
            n += 1

    def decrease_epsilon(self):
        if self.epsilon > 0.0001:
            self.epsilon -= self.epsilon_decay


class DeepQNetwork:
    def __init__(self, weights_file, learning_rate):
        self.model = self.build_model(state_size=2, action_size=2, learning_rate=learning_rate)
        self.weights_file = weights_file

        if os.path.exists(self.weights_file):
            self.model.load_weights(self.weights_file)

    def build_model(self, state_size, action_size, learning_rate):
        model = Sequential()
        model.add(Input(shape=(state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model

    def rewards_for(self, state):
        return self.model(state.reshape((1, 2)), training=False)[0].numpy()

    def save_rewards_for(self, state, rewards):
        self.model.fit(
            np.array(state),
            np.array(rewards),
            epochs=1,
            verbose=0
        )

    def update_weights(self, network):
        self.model.set_weights(network.model.get_weights())

    def save_weights(self):
        self.model.save_weights(self.weights_file)
