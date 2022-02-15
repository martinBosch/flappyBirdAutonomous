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

        self.alpha = 0.7
        self.gamma = 0.6
        self.epsilon = 0.0

        self.deep_q_network = DeepQNetwork()
        self.memory = deque(maxlen=2000)

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
        q_value = self.deep_q_network.rewards_for(state)
        # print("Q Value:", q_value)

        if np.random.uniform() < self.epsilon:
            random_action = np.random.randint(0, 2)  # Random action (do nothing, flap)
            # print("random action:", random_action)
            return random_action
        else:
            print("State:", state)
            print("Q value:", q_value)
            if q_value[FLAP] > q_value[DO_NOTHING]:
                # print("FLAP")
                return FLAP
            else:
                # print("DO NOTHING")
                return DO_NOTHING

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            immediate_reward = reward
            future_reward = self.gamma * max(self.deep_q_network.rewards_for(next_state))
            rewards_prediction = self.deep_q_network.rewards_for(state)
            rewards_prediction[action] = (1 - self.alpha) * rewards_prediction[action] + self.alpha * (immediate_reward + future_reward)
            self.deep_q_network.save_rewards_for(state, rewards_prediction)

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

        if y > 113:
            y = 113

        if y < 10:
            y = 0

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
        batch_size = 32

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

            self.remember(state=state, action=action, reward=immediate_reward, next_state=new_state, done=self.flappybird.dead)

            if self.epsilon > 0.01:
                self.epsilon -= 0.01

            if self.flappybird.dead:
                self.replay(sample_batch_size=batch_size)
                self.flappybird.restart_game()


class DeepQNetwork:
    def __init__(self):
        self.model = self.deep_q_network(state_size=2, action_size=2)

    def deep_q_network(self, state_size, action_size):
        model = Sequential()
        model.add(Input(shape=(state_size,)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

        return model

    def rewards_for(self, state):
        return self.model.predict(state.reshape((1, 2)))[0]

    def save_rewards_for(self, state, rewards):
        self.model.fit(state.reshape((1, 2)), rewards.reshape((1, 2)), epochs=1, verbose=0)
