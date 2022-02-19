import os.path
import random
import math
from time import sleep

import numpy as np
from pprint import pprint
from flappybird import FlappyBird

from collections import deque

from keras.models import Model, Sequential, Input
from keras.layers import Dense, Embedding, Reshape
from keras.optimizer_v2.adam import Adam

DO_NOTHING = 0
FLAP = 1


class Agent:
    def __init__(self):
        self.flappybird = FlappyBird(self)

        self.alpha = 0.7
        self.gamma = 0.9
        self.epsilon = 0.4

        self.q_network = DeepQNetwork(weights_file="q_network_weights.h5")
        # self.target_network = DeepQNetwork(weights_file="target_network_weights.h5")
        self.memory = deque(maxlen=2000)

        # self.target_network.update_weights(self.q_network)

    def init(self):
        pass

    def save(self):
        self.q_network.save_weights()
        # self.target_network.save_weights()

    """
     * Method used to determine the next move to be performed by the agent.
     * now is moving random
     """

    def act(self, state):
        action = self.get_action_by(state)
        if action == FLAP:
            # print("FLAP")
            self.flappybird.hold_key_down()
        else:
            # print("DO NOTHING")
            self.flappybird.release_key()

        return action

    def get_action_by(self, state):
        if np.random.uniform() < self.epsilon:
            print(f"Taking a randon value, with an epsilon ${self.epsilon}")
            return self.decide_if_flap_or_do_nothing(np.random.uniform(), np.random.uniform())
        else:
            q_value = self.q_network.rewards_for(state)
            #  print("Q Value:", q_value)
            return self.decide_if_flap_or_do_nothing(q_value[FLAP], q_value[DO_NOTHING])

    def decide_if_flap_or_do_nothing(self, q_value_flap, q_value_do_nothing):
        # print("State:", state)
        # print("Q value:", q_value)
        if q_value_flap > q_value_do_nothing:
            # print("FLAP")
            return FLAP
        else:
            # print("DO NOTHING")
            return DO_NOTHING

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            print("Lack of games for training")
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            immediate_reward = reward
            future_reward = self.gamma * max(self.q_network.rewards_for(next_state))
            rewards_prediction = self.q_network.rewards_for(state)
            print("Q values:", rewards_prediction)
            rewards_prediction[action] = (1 - self.alpha) * rewards_prediction[action] + self.alpha * (immediate_reward + future_reward)

            print("State:", state)
            print("Action:", action)
            print("Done:", done)
            print("Immediate reward:", immediate_reward)
            print("Future reward:", future_reward)
            print("Q values:", rewards_prediction)

            print("")
            print("-------")
            print("")
            self.q_network.save_rewards_for(state, rewards_prediction)

    def observe_world(self):
        positions = self.flappybird.get_world_position_objets()

        bottom_pipe = positions[0]
        upper_pipe = positions[1]
        bird = positions[2]
        print("Bottom pipe: ", bottom_pipe)
        print("Upper pipe: ", upper_pipe)
        print("Bird: ", bird)
        print("Count: ", self.flappybird.counter)
        print("Dead: ", self.flappybird.dead)

        print("x:", bottom_pipe[0] - (bird[0] + bird[2]))
        print("y:", bottom_pipe[1] - bird[1])

        print("")

    def state(self):
        positions = self.flappybird.get_world_position_objets()
        bottom_pipe = positions[0]
        bird = positions[2]

        x = bottom_pipe[0] - (bird[0] + bird[2])
        y = bottom_pipe[1] - bird[1]

        # if y > 130:
        #    y = 130

        # if y < 10:
        #    y = 0

        return np.array([x, y])

    def reward(self):
        if not self.flappybird.dead:
            # print("reward: 1")
            return 0.1
        else:
            # print("reward: -1000")
            return -100

    def run(self):
        self.flappybird.init_game()
        batch_size = 16

        n = 0
        while True:
            # sleep(0.1)
            # self.observe_world()
            state = self.state()
            # print("state:", state)
            action = self.act(state)
            # print("action:", action)

            self.flappybird.each_cycle()

            new_state = self.state()
            immediate_reward = self.reward()

            self.remember(state=state, action=action, reward=immediate_reward, next_state=new_state,
                          done=self.flappybird.dead)

            self.decrease_epsilon()

            # if self.must_update_target_network_weights(n):
            #    self.target_network.update_weights(self.q_network)

            if self.flappybird.dead:
                self.replay(sample_batch_size=batch_size)
                self.flappybird.restart_game()

            n += 1

    def must_update_target_network_weights(self, n):
        return n % 100 == 0

    def decrease_epsilon(self):
        if self.epsilon > 0.0:
            self.epsilon -= 0.01


class DeepQNetwork:
    def __init__(self, weights_file):
        self.model = self.build_model(state_size=2, action_size=2)
        self.weights_file = weights_file

        if os.path.exists(self.weights_file):
            self.model.load_weights(self.weights_file)

    def build_model(self, state_size, action_size):
        model = Sequential()
        model.add(Input(shape=(state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

        return model

    def rewards_for(self, state):
        return self.model.predict(state.reshape((1, 2)))[0]

    def save_rewards_for(self, state, rewards):
        self.model.fit(state.reshape((1, 2)), rewards.reshape((1, 2)), epochs=1, verbose=0)

    def update_weights(self, network):
        self.model.set_weights(network.model.get_weights())

    def save_weights(self):
        self.model.save_weights(self.weights_file)
