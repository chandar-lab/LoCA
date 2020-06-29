import os
import numpy as np
import json
from muzero.env import muzero_config
import torch
from torch.utils.tensorboard import SummaryWriter


def create_filename(args):
    filename_extension = ''
    if args.no_pre_training is True:
        filename_extension += '_no_pre_training'
    if args.flipped_terminals:
        filename_extension += '_flipped_terminal'
    if args.flipped_actions:
        filename_extension += '_flipped_actions'

    return args.method + filename_extension


def save_results(settings, filename, avg_perf):
    # Store results + some essential settings

    avg_performance = np.mean(np.reshape(avg_perf[:, 0, :], (settings['num_runs'], -1)), axis=0)
    std_performance = np.std(np.reshape(avg_perf[:, 0, :], (settings['num_runs'], -1)), axis=0)

    # eval_steps = []
    # settings = {}
    # settings['method'] = experiment_settings['method']
    # settings['num_steps'] = experiment_settings['num_test_steps']
    # settings['num_datapoints'] = experiment_settings['num_datapoints']
    # settings['num_runs'] = experiment_settings['num_runs']
    # settings['eval_steps'] = eval_steps

    print("file: ", filename)
    
    with open('results/' + settings['env'] + '/' + settings['method'] + '/' + filename + '_settings.txt', 'w') as json_file:
        json.dump(settings, json_file)
    np.save('results/' + settings['env'] + '/' + settings['method'] + '/' + filename + '_avg_results.npy', avg_performance)
    np.save('results/' + settings['env'] + '/' + settings['method'] + '/' + filename + '_std_results.npy', std_performance)

    # window_size = settings['num_test_steps'] // settings['num_datapoints']
    # steps = np.arange(1, settings['num_datapoints'] + 1) * window_size
    np.save('results/' + settings['env'] + '/' + settings['method'] + '/' + filename + '_eval_steps.npy',
            np.mean(np.reshape(avg_perf[:, 1, :], (settings['num_runs'], -1)), axis=0))


def update_summary_writer(config, opr, domain_settings):
    config.opr = opr
    config.seed = config.seed
    exp_path = muzero_config.set_config(config, domain_settings)
    summary_writer = SummaryWriter(exp_path, flush_secs=10)

    return summary_writer




