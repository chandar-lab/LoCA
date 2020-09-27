import os
import numpy as np
import json
from muzero.env import muzero_config
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
    """
        This function saves the performance results
        Args:
            settings: Experiment settings
            filename: the method_name + flipped_actions or not
            avg_perf: the performance matrix shape: (number of runs, 2, number of evaluation steps)
        """
    # Store results + some essential settings
    avg_performance = np.mean(np.reshape(avg_perf[:, 0, :], (settings['num_runs'], -1)), axis=0)
    std_performance = np.std(np.reshape(avg_perf[:, 0, :], (settings['num_runs'], -1)), axis=0)

    print("file saved: ", filename)

    np.save('results/' + settings['env'] + '/' + settings['method'] + '/' + filename + '_avg_results.npy', avg_performance)
    np.save('results/' + settings['env'] + '/' + settings['method'] + '/' + filename + '_std_results.npy', std_performance)
    np.save('results/' + settings['env'] + '/' + settings['method'] + '/' + filename + '_eval_steps.npy',
            np.mean(np.reshape(avg_perf[:, 1, :], (settings['num_runs'], -1)), axis=0))


def update_summary_writer(config, phase, domain_settings):
    """
        This function updates the summary writer for MuZero results
        Args:
            config: agent configs
            phase: {pre_train, training}
            domain_settings:
        Returns:
            updated summary_writer
        """
    config.opr = phase
    config.seed = config.seed
    exp_path = muzero_config.set_config(config, domain_settings)
    summary_writer = SummaryWriter(exp_path, flush_secs=10)

    return summary_writer




