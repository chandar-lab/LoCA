import os
from utils import update_summary_writer
from muzero.train import train, test
from muzero.env import muzero_config
from muzero.utils import init_logger, make_results_dir
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import math
import ray

ray.init()


class MuZeroAgent(object):
    def __init__(self, config, domain_setting, summary_writer):
        self.domain_settings = domain_setting
        # self.domain = domain
        self.config = config
        self.summary_writer = summary_writer
        self.epsilon = 0.1
        self.epsilon_init = 0.1
        self.episode_count = 0
        self.step_counter = 0
        self.fixed_behavior = False
        self.network = config.get_uniform_network()


    @staticmethod
    def print_name():
        print('SarsaLambda')

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def reset_epsilon(self):
        self.epsilon = self.epsilon_init

    def run_pretrain(self, experiment_settings, include_transition=True):
        print("train phase started.")
        self.step_counter = 0
        self.domain_settings['phase'] = 'train'
        self.config.training_steps = experiment_settings['num_train_steps']
        self.config.evaluate_interval = self.config.training_steps // experiment_settings['num_datapoints']
        self.network = train(self.config, self.domain_settings, self.network, self.summary_writer)
        if include_transition is True:
            self.config.training_steps = experiment_settings['num_transition_steps']
            self.config.evaluate_interval = self.config.training_steps // experiment_settings[
                'num_datapoints']
            summary_writer = update_summary_writer(self.config, 'transition', self.domain_settings)
            self.summary_writer = summary_writer
            self.run_local_pretrain()

    def run_local_pretrain(self):
        print("transition phase started.")
        self.domain_settings['phase'] = 'transition'
        self.network = train(self.config, self.domain_settings, self.network, self.summary_writer)

    def run_train(self, experiment_settings):
        self.domain_settings['phase'] = 'test'
        self.config.training_steps = experiment_settings['num_test_steps']
        self.config.evaluate_interval = self.config.training_steps // experiment_settings[
            'num_datapoints']
        _, performance = test(self.config, self.domain_settings, self.network, self.summary_writer)

        return np.array(performance[0]), performance[1]


def plot_state_map(init_states, terminals, phase='transition'):
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


def plot_total_return(G_training, label='test'):
    plt.plot(np.convolve(G_training, np.ones((1,)) / 1, mode='same'), label=label)
    plt.legend()
    plt.show()


def launch_job(f, *args):
  f(*args)


def build_agent(domain_settings, args):
    # set config as per arguments
    exp_path = muzero_config.set_config(args, domain_settings)
    exp_path, log_base_path = make_results_dir(exp_path, args)
    muzero_config.seed = args.seed
    muzero_config.opr = args.opr
    # set-up logger
    init_logger(log_base_path)
    summary_writer = SummaryWriter(exp_path, flush_secs=10)
    my_agent = MuZeroAgent(muzero_config, domain_settings, summary_writer)

    return my_agent


def load_agent(muzero_args, domain_settings, experiment_settings):
    muzero_args.seed = muzero_args.seed
    muzero_args.opr = 'transition'
    muzero_config.set_config(muzero_args, domain_settings)
    assert os.path.exists(muzero_config.model_path), 'model not found at {}'.format(muzero_config.model_path)
    model = muzero_config.get_uniform_network().to('cpu')
    model.load_state_dict(torch.load(muzero_config.model_path, map_location=torch.device('cpu')))
    summary_writer = update_summary_writer(muzero_args, 'test', domain_settings)
    my_agent = MuZeroAgent(muzero_config, domain_settings, summary_writer)
    my_agent.network = model
    return my_agent, muzero_config
