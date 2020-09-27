import numpy as np
import math
import random


class MountainCar_SARSA:
    def __init__(self, settings):
        self.num_actions = 3
        self.gamma = settings['gamma']
        self.num_tilings = settings['num_tilings']
        self.num_x_tiles = settings['num_x_tiles']
        self.num_v_tiles = settings['num_v_tiles']
        self.max_steps_per_episode = settings['max_steps_per_episode']
        self.num_active_features = self.num_tilings
        self.num_total_features = self.num_tilings * self.num_x_tiles * self.num_v_tiles
        self.x_range = [-1.2, 0.5]
        self.v_range = [-0.07, 0.07]
        self.tiling_x_offset = np.zeros(self.num_tilings)
        self.tiling_v_offset = np.zeros(self.num_tilings)
        self.current_state = {}

        self.init_state = settings['init_state_train']
        self.init_state_train = settings['init_state_train']
        self.init_state_transition = settings['init_state_transition']
        self.init_state_test = settings['init_state_test']
        self.init_state_eval = settings['init_state_eval']

        self.reward_terminal2_x_range = [-0.6, -0.44]
        self.reward_terminal2_v_range = [-0.003, 0.003]
        self.terminal2_radius = 0.07
        self.init = {}
        self.task = 0
        self.reward_terminal1 = 4
        self.reward_terminal2 = 2
        self.phase = 'pre_train'
        self.flipped_terminal = settings['flipped_terminals']
        self.flipped_actions = settings['flipped_actions']

        self.initialize()
        self.reset_state()

    def initialize(self):
        """
        This function can be used to initialize the domain, for example
        by randomizing the offset of the tilings.
        For our experiment, we used fixed tile positions. The reason is that
        because we only use 3 tilings, randomization can cause huge variances.
        with some representations being very bad (f.e., if all three tilings
        have similar offset).
            """

        x_tile_size = (self.x_range[1] - self.x_range[0]) / float(self.num_x_tiles)
        v_tile_size = (self.v_range[1] - self.v_range[0]) / float(self.num_v_tiles)

        for t in range(self.num_tilings):
            self.tiling_x_offset[t] = random.uniform(0, x_tile_size)
            self.tiling_v_offset[t] = random.uniform(0, v_tile_size)

    def get_domain_info(self):
        return [self.num_total_features, self.num_active_features, self.num_actions, self.gamma]

    def set_task(self, task):
        """
            This function sets the task
            terminal 1 is the terminal at the top of the hill
            terminal 2 is the terminal at the bottom of the hill
            Task A -> Task 0,
            Task B -> Task 1.
            """
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
        """
            This function sets the initial state at each phase of the experiment
            """
        self.phase = phase
        if phase == 'pre_train':
            self.init_state = self.init_state_train
        elif phase == 'local_pre_train':
            self.init_state = self.init_state_transition
        elif phase == 'train':
            self.init_state = self.init_state_test
        else:
            assert False, 'incorrect identifier'

    def set_eval_mode(self, eval):
        """
            Set the terminals with higher reward 1 and lower reward 0. It makes it easier
            to calculate evaluation performance
            The initial state needs to be adjusted as well
                """
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

    def _update_state(self, action):
        """
            This function implements dynamics
            if self.flipped_terminal is True: flip the actions to cancel the effect of model learning
                """

        if self.flipped_actions:
            action = (action + 1) % 3

        v = self.current_state['v']
        x = self.current_state['x']

        if not self.flipped_terminal:
            if x >= 0.4 and v >= 0:
                action = 2
        else:
            # if ((x + 0.52) ** 2 + 100 * v ** 2) <= 0.0164:
            if self.init_state_transition[0][0][0] <= x <= self.init_state_transition[0][0][1]\
                    and abs(v) <= self.init_state_transition[0][1][1]:
                action = 0 if v > 0 else 2

        term = 0
        next_v = v + 0.001 * (action - 1) - 0.0025 * math.cos(3 * x)
        if next_v < self.v_range[0]:
            next_v = self.v_range[0]
        elif next_v > self.v_range[1]:
            next_v = self.v_range[1]

        next_x = x + next_v
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

        self.current_state['x'] = next_x
        self.current_state['v'] = next_v
        self.current_state['terminal'] = term

    def reset_state(self):
        """
            This function randomly generate the initial state. With prob = p, it takes the init_State from the first
            interval and with prob = 1 - p from the second interval.
            This scheme should be adjusted based on the domain such that during pre-training Task A can be fully solved
            and during training Task B. And Evaluation area should be chosen such that after convergence, all the samples
            taken from that ends at terminal with higher reward!
            Return:
                 current state: {'x', 'v', 'terminal'}
                """
        reset = False
        x, v = 0, 0
        p = random.uniform(0, 1)
        while not reset:
            if len(self.init_state) == 1:
                x = random.uniform(self.init_state[0][0][0], self.init_state[0][0][1])
                v = random.uniform(self.init_state[0][1][0], self.init_state[0][1][1])
            elif len(self.init_state) == 2:  # a mix distribution for initial states
                if p < 1:
                    x = random.uniform(self.init_state[0][0][0], self.init_state[0][0][1])
                    v = random.uniform(self.init_state[0][1][0], self.init_state[0][1][1])
                else:
                    x = random.uniform(self.init_state[1][0][0], self.init_state[1][0][1])
                    v = random.uniform(self.init_state[1][1][0], self.init_state[1][1][1])

            # if (x <= self.reward_terminal2_x_range[0] or x >= self.reward_terminal2_x_range[1]) or\
            #         (v <= self.reward_terminal2_v_range[0] or v >= self.reward_terminal2_v_range[1]):
            if ((x + 0.5234) ** 2 + 100 * v ** 2) >= (self.terminal2_radius)**2:
                reset = True
        self.init['x'] = x
        self.init['v'] = v
        self.current_state['x'] = x
        self.current_state['v'] = v
        self.current_state['terminal'] = 0
        return self.current_state['x'], self.current_state['v']

    def get_state_features(self, state_features):
        self.get_active_state_features(state_features)
        return self.current_state['terminal']

    def get_active_state_features(self, state_features):
        x_size = (self.x_range[1] - self.x_range[0]) / float(self.num_x_tiles - 1)
        v_size = (self.v_range[1] - self.v_range[0]) / float(self.num_v_tiles - 1)

        for t in range(self.num_active_features):
            x = self.current_state['x'] + self.tiling_x_offset[t]
            v = self.current_state['v'] + self.tiling_v_offset[t]

            fx = int(math.floor((x - self.x_range[0]) / float(x_size)))
            fx = min(fx, self.num_x_tiles)
            fv = int(math.floor((v - self.v_range[0]) / float(v_size)))
            fv = min(fv, self.num_v_tiles)

            ft = fx + self.num_x_tiles * fv + t * self.num_x_tiles * self.num_v_tiles
            assert (0 <= ft < self.num_total_features)
            state_features[t] = ft

    def take_action(self, action, next_state_features):
        assert (0 <= action < self.num_actions)
        self._update_state(action)
        self.get_active_state_features(next_state_features)
        reward = 0
        if self.current_state['terminal'] == 1:
            reward = self.reward_terminal1
        if self.current_state['terminal'] == 2:
            reward = self.reward_terminal2
        return [reward, self.current_state['terminal']]
