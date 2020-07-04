import os
import numpy as np
import random
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from .settings import get_sarsa_settings
from sarsa_lambda.mountain_car import MountainCar_SARSA
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

    def run_pretrain(self, experiment_settings, include_transition=False):
        """
        This function runs pre-training phase on Task A (0) which includes local pre-training on task B (1)
        in the case of non-shuffled actions
        It also saves the agent after the end of local pre-training phase
        Args:
            experiment_settings
            include_transition

        """
        self.initialize()
        self.train_max_steps = experiment_settings['num_train_steps']
        self._pretrain(task=0, phase='pre_train', eval=True)
        # if include_transition:
        #     self.train_max_steps = experiment_settings['num_transition_steps']
        #     self._pretrain(task=1, phase='local_pre_train', eval=False)

        if experiment_settings['save']:
            os.makedirs('results/' + experiment_settings['env'] + '/sarsa_lambda/agents/', exist_ok=True)
            with open('results/' + experiment_settings['env'] + '/sarsa_lambda/agents/' + experiment_settings['filename'] + '.pkl', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        self.domain.flipped_actions = False
        
    def run_train(self, experiment_settings):
        """
            This function runs training phase on Task B (1) and the evaluation
            It also can plot the heatmap of state space of the domain for the number of steps each states takes to reach
             the terminal or which termianl it is ended yp.
            Args:
                experiment_settings

            """
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase('train')
        self.train_max_steps = experiment_settings['num_test_steps']
        self.set_epsilon(self.epsilon_init_test)
        self.epsilon_init = self.epsilon_init_test
        self.epsilon_decay_steps = self.epsilon_decay_steps_test

        if self.theta_train is not None:
            self.theta = np.copy(self.theta_train)

        print(''), print('reward terminal 1: {}, reward terminal 2: {}'.format(self.domain.reward_terminal1,
                                                                               self.domain.reward_terminal2))
        terminals_test, init_states, G_test, perf_list = [], [], [], []
        num_steps, num_episodes = 0, 0
        self.step_counter, self.episode_count = 0, 0
        performance = []
        eval = True
        while num_steps < self.train_max_steps:
            [G, n_steps, terminal, init_state, perf] = self._run_episode(num_datapoints=experiment_settings['num_datapoints'],
                                                                        eval=eval, fixed_behavior=self.fixed_behavior) #
            # self.decay_epsilon()
            terminals_test.append(terminal)
            init_states.append(init_state)
            G_test.append(G)
            self.episode_count += 1
            if self.episode_count % 50 == 0:
                print('Step: {} ] G: {:<8.3f} ] Episodes Collected: {:<10d}, epsilon: {}'.format(
                    num_steps, sum(G_test[-20:]) / 20, self.episode_count, self.epsilon))
            for p in perf:
                performance.append(p)
            num_steps += n_steps

        plot_state_map(init_states, terminals_test, phase='test')
        if eval:
            plot_total_return(performance, label='test')
            print("avg reward: ", sum(performance) / len(performance), " final reward: ",
                  np.mean(np.array(performance[-3:])))

        self.episode_count = 0
        window_size = self.train_max_steps // experiment_settings['num_datapoints']
        steps = np.arange(1, experiment_settings['num_datapoints'] + 1) * window_size
        return np.array(performance), steps

    def _pretrain(self, task, phase, eval):
        """
            This function runs pre-training phase on a given task and the evaluation
            It also can plot the heatmap of state space of the domain for the number of steps each states takes to reach
             the terminal or which termianl it is ended yp.
            Args:
                experiment_settings

            """
        print("{} phase started.".format(phase))
        self.domain.set_task(task)  # set domain to task A
        self.domain.set_phase(phase)
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
            [G, n_steps, terminal, init_state, perf] = self._run_episode(num_datapoints=10, eval=eval)
            self.decay_epsilon()

            G_training.append(G)

            terminals_training.append(terminal)
            init_states.append(init_state)
            num_steps += n_steps
            self.episode_count += 1
            if phase is 'pre_train' and self.episode_count % 500 == 0:
                print('Step: {} ] G: {:<8.3f} ] Episodes Collected: {:<10d} ] epsilon: {}'.format(
                    num_steps, sum(G_training[-20:]) / 20, self.episode_count, self.epsilon))

            for p in perf:
                performance.append(p)

        # plot the state map
        plot_state_map(init_states, terminals_training, phase=phase)
        plot_total_return(G_training, label=phase)
        if eval:
            plot_total_return(performance, label=phase)

        print('Pre-Training: Percentage of episodes ended at terminal A :{} %, terminal B: {} %'.format(
            terminals_training.count(1) / len(terminals_training),
            terminals_training.count(2) / len(terminals_training)))
        self.episode_count = 0
        self.theta_train = self.theta

    def _run_episode(self, num_datapoints=10, eval=False, fixed_behavior=False):
        """
            This function runs one episode and runs evaluations at window size intervals
            Args:
                num_datapoints: number of evaluation steps
                eval: Run evaluation or not
            Return:
                G: Discounted return of the episode
                num_steps: number of steps of each episode,
                terminal: which terminal is ended,
                init_state: Episode started at this state
                performance: Evaluation performance (Top terminal fraction between 0 and 1)

            """
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
        """
            This function runs the evaluation eval_episodes times and take average
            Basically it count how many times it ends at higher reward terminal and calculates the fraction
            Return:
                average performance
            """
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

        performance = sum_rewards/self.eval_episodes
        self.domain.set_eval_mode(False)
        self.set_epsilon(epsilon)
        # print('Evaluation Done!')
        return performance

    def decay_epsilon(self):
        epsilon = self.epsilon_init * self.ep_decay_rate ** (self.episode_count / self.epsilon_decay_steps)
        self.epsilon = max(epsilon, 0.01)


def plot_state_map(init_states, terminals, phase='transition'):
    """
        This function gets the list of initial states and terminals each episode ended and plots the heatmap
        Args:
            init_states: list
            terminals = list
        """
    sns.set()
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


def plot_total_return(return_curve, label='pre-training'):
    """
       This function gets the list of the discounted returns and plot the curve
       Args:
           return_curve: list
           label: {pre-training, training}
       """
    plt.plot(np.convolve(return_curve, np.ones((1,)) / 1, mode='same'), label=label)
    plt.legend()
    plt.show()


def build_agent(domain_settings, args):
    """
        This function gets the agent's settings and domain's setting and build the sarsa_lambda agent
        Args:
            domain_settings
        Return:
            sarsa_lambda agent
        """
    agent_settings = get_sarsa_settings()
    my_domain = MountainCar_SARSA(domain_settings)
    my_agent = SarsaLambdaAgent(agent_settings, my_domain)

    return my_agent


def load_agent(agent_args, domain_settings, experiment_settings):
    """
        This function loads the agent from the results directory results/env_name/method_name/filename
        Args:
            experiment_settings
        Return:
            sarsa_lambda agent
        """
    with open('results/' + experiment_settings['env'] + '/sarsa_lambda/agents/' + experiment_settings['filename'] + '.pkl', 'rb') as input:
        my_agent = pickle.load(input)

    return my_agent, None

