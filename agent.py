import random
import math
from time import sleep

import numpy as np
from flappybird import FlappyBird

DO_NOTHING = 0
FLAP = 1


class Agent:
    def __init__(self):
        self.flappybird = FlappyBird()
        self.q_table = {}

    def init(self):
        pass

    """
     * Method used to determine the next move to be performed by the agent.
     * now is moving random
     """
    def act(self, epsilon, state):
        action = self.action(epsilon, state)
        if action == FLAP:
            self.flappybird.hold_key_down()
        else:
            self.flappybird.release_key()

        return action

    def action(self, epsilon, state):
        q_value = self.q_table.setdefault(state, [1, 0])

        if np.random.uniform() < epsilon:
            random_action = np.random.randint(0, 2)  # Random action (do nothing, flap)
            # print("random action:", random_action)
            return random_action
        else:
            # print("State:", state)
            # print("Q value:", q_value)
            if q_value[FLAP] > q_value[DO_NOTHING]:
                self.flappybird.hold_key_down()
                # print("FLAP")
                return FLAP
            else:
                self.flappybird.release_key()
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

        # if y < 10:
        #    y = -1

        return x, y

    def reward(self):
        if not self.flappybird.dead:
            # print("reward: 1")
            return 1
        else:
            # print("reward: -1000")
            return -1000

    def run(self):
        self.flappybird.init_game()
        epsilon = 0.0
        while True:
            # sleep(0.5)
            # self.observe_world()
            state = self.state()
            # print("state:", state)
            action = self.act(epsilon, state)
            # print("action:", action)

            self.flappybird.each_cycle()

            new_state = self.state()

            # Q Learning
            gamma = 0.9
            alpha = 0.7

            immediate_reward = self.reward()
            future_reward = gamma * max(self.q_table.get(new_state, [1, 0]))
            self.q_table[state][action] = (1 - alpha) * self.q_table.get(state)[action] + alpha * (immediate_reward + future_reward)

            # print("Q Table:", self.q_table.get(state))

            if epsilon > 0.01:
                epsilon -= 0.01
            if self.flappybird.dead:
                self.flappybird.restart_game()
