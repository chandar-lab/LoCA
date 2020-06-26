"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np
import random
import torch

import gym
from gym import spaces
from gym.utils import seeding
from ..mcts import Node
from ..game import Action, ActionHistory
from collections import deque



class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, settings, seed=1, goal_velocity=0):

        self.child_visits = []
        self.root_values = []
        self.discount = settings['gamma']
        self.G = 0

        self.k = 3
        self.frames = deque([], maxlen=self.k)

        self.x_range = [-1.2, 0.5]
        self.v_range = [-0.07, 0.07]

        self.goal_position = 0.05
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.x_range[0], self.v_range[0]], dtype=np.float32)
        self.high = np.array([self.x_range[1], self.v_range[1]], dtype=np.float32)

        self.num_actions = 3
        self.gamma = settings['gamma']
        self.init_state = settings['init_state_train']
        self.init_state_train = settings['init_state_train']
        self.init_state_transition = settings['init_state_transition']
        self.init_state_test = settings['init_state_test']
        self.init_state_eval = settings['init_state_eval']

        self.reward_terminal2_x_range = [-0.6, -0.44]
        self.reward_terminal2_v_range = [-0.003, 0.003]
        self.terminal2_radius = 0.07

        self.current_state = {}
        self.init = {}
        self.task = 0
        self.reward_terminal1 = 4
        self.reward_terminal2 = 2
        self.phase = settings['phase']
        self.flipped_terminal = settings['flipped_terminals']
        self.flipped_actions = settings['flipped_actions']


        # self.states = [self.reset()]
        self.action_space = spaces.Discrete(3)
        self.action_space_size = 3
        self.actions = list(map(lambda i: Action(i), range(self.action_space.n))) #range(3) #
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_task(self, task):
        self.task = task
        if self.flipped_terminal:
            if task == 0:  # taskA
                self.reward_terminal1 = 2
                self.reward_terminal2 = 4
            else:  # taskB
                self.reward_terminal1 = 2
                self.reward_terminal2 = 1
        else:
            if task == 0:  # taskA
                self.reward_terminal1 = 4
                self.reward_terminal2 = 2
            else:  # taskB
                self.reward_terminal1 = 1
                self.reward_terminal2 = 2

    def set_phase(self, phase):
        self.phase = phase
        if phase == 'train':
            self.init_state = self.init_state_train
        elif phase == 'transition':
            self.init_state = self.init_state_transition
        elif phase == 'test':
            self.init_state = self.init_state_test
        else:
            assert False, 'incorrect identifier'

    def set_eval_mode(self, eval):
        if eval:
            if self.reward_terminal1 > self.reward_terminal2:
                self.reward_terminal1 = 1
                self.reward_terminal2 = 0
            else:
                self.reward_terminal1 = 0
                self.reward_terminal2 = 1
            self.init_state = self.init_state_eval
        else:
            self.set_task(self.task)
            self.set_phase(self.phase)

    def step(self, action):
        # assert self.actions.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.flipped_actions:
            action = (action + 1) % 3

        v = self.current_state['v']
        x = self.current_state['x']

        if not self.flipped_terminal:
            if x >= 0.4 and v >= 0:
                action = 2
        else:
            # if ((x + 0.52) ** 2 + 100 * v ** 2) <= 0.0164:
            if self.init_state_transition[0][0][0] <= x <= self.init_state_transition[0][0][1] \
                    and abs(v) <= self.init_state_transition[0][1][1]:
                action = 0 if v > 0 else 2

        next_v = v + self.force * (action - 1) - self.gravity * math.cos(3 * x)

        if next_v < self.v_range[0]:
            next_v = self.v_range[0]
        elif next_v > self.v_range[1]:
            next_v = self.v_range[1]

        next_x = x + next_v
        term = 0
        if next_x <= self.x_range[0]:
            next_x = self.x_range[0]
            next_v = 0
        elif next_x >= self.x_range[1]:
            next_x = self.x_range[1]
            next_v = 0
            term = 1
        # elif self.reward_terminal2_x_range[0] <= next_x <= self.reward_terminal2_x_range[1]\
        #         and abs(next_v) <= self.reward_terminal2_v_range[1]:
        elif ((next_x + 0.52) ** 2 + 100 * next_v ** 2) <= (self.terminal2_radius)**2:
            next_v = 0
            term = 2

        self.current_state['terminal'] = term
        self.current_state['x'] = next_x
        self.current_state['v'] = next_v
        self.states += [np.array([self.current_state['x'], self.current_state['v']])]
        reward = 0
        if self.current_state['terminal'] == 1:
            reward = self.reward_terminal1
        if self.current_state['terminal'] == 2:
            reward = self.reward_terminal2

        self.rewards.append(reward)
        self.history.append(action)

        return self.obs(len(self.rewards)), reward, term, {}

    def reset(self):
        self.history = []
        self.rewards = []
        self.states = []

        reset = False
        x, v = 0, 0
        p = random.uniform(0, 1)
        while not reset:
            if len(self.init_state) == 1:
                x = random.uniform(self.init_state[0][0][0], self.init_state[0][0][1])
                v = random.uniform(self.init_state[0][1][0], self.init_state[0][1][1])
            elif len(self.init_state) == 2:  # a mix distribution for initial states
                if p < 0.5:
                    x = random.uniform(self.init_state[0][0][0], self.init_state[0][0][1])
                    v = random.uniform(self.init_state[0][1][0], self.init_state[0][1][1])
                else:
                    x = random.uniform(self.init_state[1][0][0], self.init_state[1][0][1])
                    v = random.uniform(self.init_state[1][1][0], self.init_state[1][1][1])

            if ((x + 0.5234) ** 2 + 100 * v ** 2) >= (self.terminal2_radius)**2:
            # if (x <= self.reward_terminal2_x_range[0] or x >= self.reward_terminal2_x_range[1]) or \
            #             (v <= self.reward_terminal2_v_range[0] or v >= self.reward_terminal2_v_range[1]):
                reset = True
        self.init['x'] = x
        self.init['v'] = v
        self.current_state['x'] = x
        self.current_state['v'] = v
        self.current_state['terminal'] = 0

        for _ in range(self.k):
            self.states.append(np.array([self.current_state['x'], self.current_state['v']]))
        return self.obs(0)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55


    def get_keys_to_action(self):
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}  # control with left and right arrow keys

    def obs(self, i: int):
        """Compute the state of the game."""
        # return self.states[state_index]
        frames = self.states[i:i + self.k]
        return np.array(frames).flatten()

    def legal_actions(self):
        """Return the legal actions available at this instant."""
        return self.actions

    def action_history(self):
        """Return the actions executed inside the search."""
        return ActionHistory(self.history, 3)

    def to_play(self):
        """Return the current player."""
        return 0

    def store_search_statistics(self, root: Node):
        """After each MCTS run, store the statistics generated by the search."""

        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in self.actions
        ])
        self.root_values.append(np.maximum(0, root.value()))

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, model=None, config=None):
        # The value target is the discounted root value of the search tree N steps into the future, plus
        # the discounted sum of all rewards until then.
        target_values, target_rewards, target_policies = [], [], []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                if model is None:
                    value = self.root_values[bootstrap_index] * self.discount ** td_steps
                else:
                    # Reference : Appendix H => Reanalyze
                    # Note : a target network  based on recent parameters is used to provide a fresher,
                    # stable n-step bootstrapped target for the value function
                    obs = self.obs(bootstrap_index)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    network_output = model.initial_inference(obs)
                    value = network_output.value.data.cpu().item() * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index < len(self.root_values):
                target_values.append(value)
                target_rewards.append(self.rewards[current_index])

                # Reference : Appendix H => Reanalyze
                # Note : MuZero Reanalyze revisits its past time-steps and re-executes its search using the
                # latest model parameters, potentially resulting in a better quality policy than the original search.
                # This fresh policy is used as the policy target for 80% of updates during MuZero training
                if model is not None and random.random() <= config.revisit_policy_search_rate:
                    from ..mcts import MCTS, Node
                    root = Node(0)
                    obs = self.obs(current_index)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    network_output = model.initial_inference(obs)
                    root.expand(self.to_play(), self.legal_actions(), network_output)
                    MCTS(config).run(root, self.action_history(), model)
                    self.store_search_statistics(root)

                target_policies.append(self.child_visits[current_index])

            else:
                # States past the end of games are treated as absorbing states.
                target_values.append(0)
                target_rewards.append(0)
                # Note: Target policy is  set to 0 so that no policy loss is calculated for them
                target_policies.append([0 for _ in range(len(self.child_visits[0]))])

        return target_values, target_rewards, target_policies

    def terminal(self):
        return self.current_state['terminal']

    def __len__(self):
        return len(self.history)
