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


def save_results(experiment_settings, filename, avg_perf):
    # Store results + some essential settings

    avg_performance = np.mean(avg_perf, axis=0)
    std_performance = np.std(avg_perf, axis=0)

    eval_steps = []
    settings = {}
    settings['method_name'] = experiment_settings['method']
    settings['num_steps'] = experiment_settings['num_test_steps']
    settings['num_datapoints'] = experiment_settings['num_datapoints']
    settings['num_runs'] = experiment_settings['num_runs']
    settings['eval_steps'] = eval_steps

    print("file: ", filename)
    Done = False
    # for i in range(len(avg_performance)):
    #     if np.mean(avg_performance[i-5:i]) > 0.95:
    #         Done = True
    #     if Done:
    #         avg_performance[i] = 1
    #         std_performance[i] = 0

    with open('data/' + filename + '_settings.txt', 'w') as json_file:
        json.dump(settings, json_file)
    np.save('data/' + filename + '_avg_results.npy', avg_performance)
    np.save('data/' + filename + '_std_results.npy', std_performance)


def update_summary_writer(config, opr, domain_settings):
    config.opr = opr
    config.seed = config.seed + 1
    exp_path = muzero_config.set_config(config, domain_settings)
    summary_writer = SummaryWriter(exp_path, flush_secs=10)

    return summary_writer




