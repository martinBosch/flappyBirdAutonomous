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
        self.flappybird = FlappyBird()
        self._state_size = 10 * 10
        self._action_size = 2
        self._optimizer = Adam(learning_rate=0.01)

        self.alpha = 0.7
        self.gamma = 0.6
        self.epsilon = 0.0

        self.q_network = self.deep_q_network()

    def deep_q_network(self):
        model = Sequential()
        model.add(Input(shape=(2,)))
        model.add(Dense(self._state_size, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=self._optimizer)

        return model

    def init(self):
        pass

    """
     * Method used to determine the next move to be performed by the agent.
     * now is moving random
     """
    def act(self, state):
        action = self.action(state)
        if action == FLAP:
            # print("FLAP")
            self.flappybird.hold_key_down()
        else:
            # print("DO NOTHING")
            self.flappybird.release_key()

        return action

    def action(self, state):
        q_value = self.q_network.predict(state.reshape((1, 2)))[0]
        print("Q Value:", q_value)

        if np.random.uniform() < self.epsilon:
            random_action = np.random.randint(0, 2)  # Random action (do nothing, flap)
            # print("random action:", random_action)
            return random_action
        else:
            # print("State:", state)
            # print("Q value:", q_value)
            if q_value[FLAP] > q_value[DO_NOTHING]:
                # print("FLAP")
                return FLAP
            else:
                # print("DO NOTHING")
                return DO_NOTHING

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

        # if x > 200:
        #    x = 200

        # if y > 113:
        #    y = 113

        # if y < 10:
        #    y = 0

        return np.array([x, y])

    def reward(self):
        if not self.flappybird.dead:
            # print("reward: 1")
            return 0.1
        else:
            # print("reward: -1000")
            return -10

    def run(self):
        self.flappybird.init_game()
        batch_size = 1

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

            future_reward = self.gamma * max(self.q_network.predict(new_state.reshape((1, 2)))[0])
            self.q_network.predict(state.reshape((1, 2)))[0][action] = (1 - self.alpha) * self.q_network.predict(state.reshape((1, 2)))[0][action] + self.alpha * (immediate_reward + future_reward)

            if self.epsilon > 0.01:
                self.epsilon -= 0.01

            if self.flappybird.dead:
                self.flappybird.restart_game()
