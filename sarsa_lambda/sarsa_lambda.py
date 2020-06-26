import numpy as np
import random
import seaborn as sns; sns.set()
from scipy import stats
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from config import get_sarsa_setting
from mountain_car import MountainCar_SARSA
import pickle


class SarsaLambdaAgent(object):
    def __init__(self, settings, domain):
        self.domain = domain
        [self.num_state_features, self.num_active_features, self.num_actions, self.gamma] = domain.get_domain_info()
        self.total_features = self.num_state_features * self.num_actions
        self.alpha = settings['alpha']
        self.lAmbda = settings['lambda']
        self.ep_decay_rate = settings['ep_decay_rate']
        self.epsilon_decay_steps = settings['epsilon_decay_steps_train']
        self.epsilon_decay_steps_test = settings['epsilon_decay_steps_test']
        self.epsilon_init = settings['epsilon_init']
        self.epsilon_init_test = settings['epsilon_init_test']
        self.epsilon = settings['epsilon_init']
        self.theta_init = settings['theta_init']
        self.eval_episodes = settings['eval_episodes']
        self.eval_epsilon = settings['eval_epsilon']
        self.eval_max_steps = settings['eval_max_steps']
        self.fixed_behavior = settings['fixed_behavior']
        self.train_max_steps = settings['train_max_steps']

        self.behavior_policy = []

        self.type = settings['type']
        self.episode_count = 0
        self.step_counter = 0

        self.theta = np.zeros(self.total_features)
        self.theta_train = None
        self.theta_transition = None
        self.e_trace = np.zeros(self.total_features)

    @staticmethod
    def print_name():
        print('SarsaLambda')

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def reset_epsilon(self):
        self.epsilon = self.epsilon_init

    def _initialize_trace(self):
        for i in range(self.total_features):
            self.e_trace[i] = 0
        self.Qs_old = 0

    def run_train(self, experiment_settings, include_transition=False):
        # Training phase
        self.initialize()
        # my_agent.set_epsilon(1.0)  # some high exploration early in training to ensure model gets learned well
        # my_agent.run_train(Experiment_settings['num_train_steps1'])
        # my_agent.reset_epsilon()
        self.train_max_steps = experiment_settings['num_train_steps']
        self._train()
        if include_transition:
            self.train_max_steps = experiment_settings['num_transition_steps']
            self._transition()

        if experiment_settings['save']:
            with open('results/sarsa_lambda/agents/' + experiment_settings['filename'] + '.pkl', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        self.domain.flipped_actions = False

    def _train(self):
        print("pre-train phase started.")
        self.domain.set_task(0)  # set domain to task A
        self.domain.set_phase('train')
        print(''), print('reward terminal 1: {}, reward terminal 2: {}'.format(self.domain.reward_terminal1,
                         self.domain.reward_terminal2))
        if self.theta_train is None:
            self.theta = np.zeros(self.total_features)
        else:
            self.theta = np.copy(self.theta_train)
        num_steps = 0
        terminals_training, G_training, init_states, performance = [], [], [], []
        self.step_counter, self.episode_count = 0,  0

        while num_steps < self.train_max_steps:
            [G, n_steps, terminal, init_state, perf] = self.run_episode(num_datapoints=100, eval=True)
            self.decay_epsilon()

            G_training.append(G)
            if self.episode_count > 500:
                terminals_training.append(terminal)
                init_states.append(init_state)
            num_steps += n_steps
            self.episode_count += 1
            if self.episode_count % 500 == 0:
                print('Step: {} ] G: {:<8.3f} ] Episodes Collected: {:<10d} ] epsilon: {}'.format(
                    num_steps, sum(G_training[-20:]) / 20, self.episode_count, self.epsilon))

            for p in perf:
                performance.append(p)

        # plot the state map
        plot_state_map(init_states, terminals_training, phase='train')
        plot_total_return(G_training, label='train')
        plot_total_return(performance, label='train')

        print('Pre-Training: Percentage of episodes ended at terminal A :{} %, terminal B: {} %'.format(
            terminals_training.count(1) / len(terminals_training),
            terminals_training.count(2) / len(terminals_training)))
        self.episode_count = 0
        self.theta_train = self.theta
        self.behavior_policy = self.get_behavior_policy()

    def _transition(self):
        print("Local pre-training phase started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase('transition')
        assert self.theta_train is not None
        self.theta = np.copy(self.theta_train)
        print(''), print('reward terminal 1: {}, reward terminal 2: {}'.format(self.domain.reward_terminal1,
                                                                               self.domain.reward_terminal2))
        terminals_transition, init_states = [], []
        self.step_counter, num_steps = 0, 0
        while num_steps < self.train_max_steps:
            [G, n_steps, terminal, init_state, _] = self.run_episode(fixed_behavior=True)
            terminals_transition.append(terminal)
            init_states.append(init_state)
            num_steps += n_steps
            if num_steps % 5000 == 0:
                print('Step: {} ] G: {:<8.3f} ] Episodes Collected: {:<10d}'.format(
                    num_steps, G, self.episode_count))

        self.theta_transition = self.theta
        self.episode_count = 0
        plot_state_map(init_states, terminals_transition, phase='transition')

    def run_test(self, experiment_settings):
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase('test')
        self.train_max_steps = experiment_settings['num_test_steps']
        self.set_epsilon(self.epsilon_init_test)
        self.epsilon_init = self.epsilon_init_test
        self.epsilon_decay_steps = self.epsilon_decay_steps_test
        if self.theta_transition is not None:
            self.theta = np.copy(self.theta_transition)
        elif self.theta_train is not None:
            self.theta = np.copy(self.theta_train)

        print(''), print('reward terminal 1: {}, reward terminal 2: {}'.format(self.domain.reward_terminal1,
                                                                               self.domain.reward_terminal2))
        terminals_test, init_states, G_test, perf_list = [], [], [], []
        num_steps, num_episodes, data_point = 0, 0, 0
        self.step_counter, self.episode_count = 0, 0
        performance = []
        while num_steps < self.train_max_steps:
            [G, n_steps, terminal, init_state, perf] = self.run_episode(num_datapoints=100,
                                                                        eval=True, fixed_behavior=self.fixed_behavior) #
            # self.decay_epsilon()

            if self.episode_count > 400:
                terminals_test.append(terminal)
                init_states.append(init_state)
            G_test.append(G)
            self.episode_count += 1
            if self.episode_count % 200 == 0:
                print('Step: {} ] G: {:<8.3f} ] Episodes Collected: {:<10d}, epsilon: {}'.format(
                    num_steps, sum(G_test[-20:]) / 20, self.episode_count, self.epsilon))
            for p in perf:
                performance.append(p)
            num_steps += n_steps

        plot_state_map(init_states, terminals_test, phase='test')
        plot_total_return(performance, label='performance')
        self.episode_count = 0
        print("avg reward: ", sum(performance) / len(performance), " final reward: ", np.mean(np.array(performance[-3:])))
        return np.array(performance), None

    def run_episode(self, num_datapoints=10, eval=False, fixed_behavior = False):
        window_size = self.train_max_steps // num_datapoints
        data_point = 0
        performance = []

        state_features = np.zeros(self.num_active_features, dtype=np.int)

        self._initialize_trace()
        init_state = self.domain.reset_state()
        terminal = self.domain.get_state_features(state_features)

        action = self.select_action(state_features)
        action_features = self._get_action_features(state_features, action, terminal)

        num_steps = 0
        G = 0.0
        total_discount = 1.0
        while (terminal == 0) and (num_steps < self.domain.max_steps_per_episode):
            [reward, terminal] = self.domain.take_action(action, state_features)
            next_action = self.select_action(state_features)
            next_action_features = self._get_action_features(state_features, next_action, terminal)
            num_steps += 1
            self.step_counter += 1
            G += total_discount * reward
            total_discount *= self.gamma

            if fixed_behavior is False:
                self._update_theta(action_features, reward, next_action_features)

            action_features = next_action_features
            action = next_action

            if eval:
                if ((self.step_counter + 1) % window_size == 0) and (data_point < num_datapoints):
                    self.domain.set_eval_mode(True)
                    performance.append(self.eval_policy())
                    data_point += 1

        return [G, num_steps, terminal, init_state, performance]

    def _update_theta(self, update_features, reward, update_features2):

        Qs = sum(self.theta[update_features])
        Qs2 = sum(self.theta[update_features2])

        delta = reward + self.gamma * Qs2 - Qs

        # # just sarsa(0)
        # self.theta[update_features] += self.alpha * delta

        if self.type == 2:
            delta += Qs - self.Qs_old

        # update traces
        if self.type == 0:  # accumulating traces
            self.e_trace[update_features] += self.alpha
        elif self.type == 1:  # replacing traces
            self.e_trace[update_features] = self.alpha
        elif self.type == 2:  # dutch traces
            e_phi = sum(self.e_trace[update_features])
            self.e_trace[update_features] += self.alpha * (1 - e_phi)
        else:
            assert False

        self.theta += self.e_trace * delta

        if self.type == 2:
            self.theta[update_features] -= self.alpha * (Qs - self.Qs_old)
            self.Qs_old = Qs2

        self.e_trace *= self.gamma * self.lAmbda

    def select_action(self, state_features):
        if self.epsilon >= 1.0:
            return random.randint(0, self.num_actions - 1)

        # determine Q-values
        Q = [0.0] * self.num_actions
        for a in range(self.num_actions):
            for j in range(self.num_active_features):
                f = state_features[j] + a * self.num_state_features
                Q[a] += self.theta[f]

        # determine Qmax & num_max of the array Q[s]
        Qmax = Q[0]
        num_max = 1
        for i in range(1, self.num_actions):
            if Q[i] > Qmax:
                Qmax = Q[i]
                num_max = 1
            elif Q[i] == Qmax:
                num_max += 1

        # simultaneously compute selection probability for each action and select action
        rnd = random.random()
        cumulative_prob = 0.0
        action = self.num_actions - 1
        for a in range(self.num_actions - 1):
            prob = self.epsilon / float(self.num_actions)
            if Q[a] == Qmax:
                prob += (1 - self.epsilon) / float(num_max)
            cumulative_prob += prob

            if rnd < cumulative_prob:
                action = a
                break

        return action

    def select_from_policy(self, state, policy):
        # rnd = np.random.random()/1.0000001
        # sum_p = 0.0
        # for a in range(self.num_actions):
        #     sum_p += policy[state][a]
        #     if rnd < sum_p:
        #         return a
        # assert False, "action not selected"
        pass

    def initialize(self):
        self.domain.initialize()
        for i in range(self.total_features):
            self.theta[i] = self.theta_init
            self.e_trace[i] = 0
        self.Qs_old = 0

    def _get_action_features(self, state_features, action, terminal):
        action_features = np.zeros(self.num_active_features, dtype=np.int)
        for i in range(self.num_active_features):
            if terminal > 0:
                action_features[i] = -1
            else:
                action_features[i] = int(state_features[i] + action * self.num_state_features)
        return action_features

    def eval_policy(self):
        epsilon = self.epsilon
        self.set_epsilon(self.eval_epsilon)
        # self._initialize_trace()
        domain = deepcopy(self.domain)
        c, performance = 0, 0
        failed_init = []

        sum_rewards = 0
        for ep in range(self.eval_episodes):
            init = domain.reset_state()
            state_features = np.zeros(self.num_active_features, dtype=np.int)
            action = self.select_action(state_features)
            terminal = domain.get_state_features(state_features)

            for i in range(self.eval_max_steps):
                [reward, terminal] = domain.take_action(action, state_features)
                next_action = self.select_action(state_features)
                action = next_action
                if terminal != 0:
                    sum_rewards += reward
                    c += 1
                    if reward == 0:
                        failed_init.append(init)
                    break
        if c > 0:
            performance = sum_rewards/self.eval_episodes
            # if len(failed_init) > 0:
            #     print(failed_init)
        self.domain.set_eval_mode(False)
        self.set_epsilon(epsilon)
        # print('Evaluation Done!')
        return performance
    #
    # def get_eval_policy(self):
    #     return self._get_egreedy_policy(self.q, self.eval_epsilon)

    def get_behavior_policy(self):
        return self._get_egreedy_policy()

    def _get_egreedy_policy(self):
        return None

    def decay_epsilon(self):
        epsilon = self.epsilon_init * self.ep_decay_rate ** (self.episode_count / self.epsilon_decay_steps)
        self.epsilon = max(epsilon, 0.01)


def plot_state_map(init_states, terminals, phase='transition'):
    state_map_vec = np.empty((140, 170, 10000))
    state_map_vec[:] = np.NaN
    x, v = list(np.round(np.linspace(-1.2, 0.5, 170), 2)), list(np.round(np.linspace(-0.07, 0.07, 140), 3))
    for i in range(len(init_states)):
        run = 0
        if np.round(init_states[i][0], 2) in x and np.round(init_states[i][1], 3) in v:
            while not math.isnan(
                    state_map_vec[v.index(np.round(init_states[i][1], 3)), x.index(np.round(init_states[i][0], 2)), run]):
                run += 1

            state_map_vec[v.index(np.round(init_states[i][1], 3)), x.index(np.round(init_states[i][0], 2)), run] = terminals[i]

    state_map = np.reshape(stats.mode(state_map_vec, axis=2, nan_policy='omit')[0], (140, 170))
    state_map = np.column_stack((state_map, np.zeros((140, 20))))
    ax = sns.heatmap(state_map)
    two_id_x = [i * 10 for i in range(1, int(np.round(len(x) / 10)))]
    two_id_v = [i * 10 for i in range(1, int(np.round(len(v) / 10)))]
    ax.set_xticks(two_id_x)
    ax.set_xticklabels(x[_two_id_x] for _two_id_x in two_id_x)
    ax.set_yticks(two_id_v)
    ax.set_yticklabels(v[_two_id_v] for _two_id_v in two_id_v)
    ax.collections[0].colorbar.ax.set_ylim(0, 2)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('State space of MountainCar during: {}'.format(phase))
    plt.show()


def plot_total_return(G_training, label='test'):
    plt.plot(np.convolve(G_training, np.ones((1,)) / 1, mode='same'), label=label)
    plt.legend()
    plt.show()


def build_agent(domain_settings, args):
    agent_settings = get_sarsa_setting()
    my_domain = MountainCar_SARSA(domain_settings)
    my_agent = SarsaLambdaAgent(agent_settings, my_domain)

    return my_agent


def load_agent(agent_args, domain_settings, experiment_settings):
    with open('results/sarsa_lambda/agents/' + experiment_settings['filename'] + '.pkl', 'rb') as input:
        my_agent = pickle.load(input)

    return my_agent, _

