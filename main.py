import numpy as np
import time

from muzero.MuZeroAgent import MuZeroAgent, build_agent
from muzero.arguments import get_muzero_args
from muzero.env import muzero_config
from utils import create_filename, save_results, update_summary_writer
from config import get_doamin_setting, get_experiment_setting

import argparse
import pickle
from copy import deepcopy

# Settings  ##########################################################################
# Lets gather arguments
parser = argparse.ArgumentParser(description='LoCA regret Experiments')
parser.add_argument('--method', default='sarsa_lambda', help='Name of the method')
parser.add_argument('--env', default='MountainCar', help='Name of the environment')
parser.add_argument('--no_pre_training',  action='store_true', default=False)
parser.add_argument('--load',  action='store_true', default=False)
parser.add_argument('--save',  action='store_true', default=False)
parser.add_argument('--flipped_terminals',  action='store_true', default=False)
parser.add_argument('--flipped_actions',  action='store_true', default=False)

args = parser.parse_args()
Experiment_settings = get_experiment_setting(args)
domain_settings = get_doamin_setting(args)
filename = create_filename(args)
print("file: ", filename)
Experiment_settings['filename'] = filename

if Experiment_settings['method'] == 1:
    agent_args = []
    from sarsa_lambda.sarsa_lambda import build_agent, load_agent
elif Experiment_settings['method'] == 2:
    agent_args = get_muzero_args()
    agent_args.flippedTask = args.flipped_terminals
    agent_args.flippedActions = args.flipped_actions
    print(agent_args.result_dir)
    from muzero.MuZeroAgent import build_agent, load_agent
else:
    assert False, 'HvS: Invalid method id.'

# Pre-Training phase ###########################################################################################
start = time.time()
my_agent = None
if not args.load:  # train
    if args.no_pre_training is False:
        my_agent = build_agent(domain_settings, agent_args)
        my_agent.run_train(Experiment_settings, include_transition=not domain_settings['flipped_actions'])
    else:  # no pre-training
        agent_args.opr = 'test'
        my_agent = build_agent(domain_settings, agent_args)
    print("time: {}s".format(time.time()-start))

else:  # load
    my_agent, agent_config = load_agent(agent_args, domain_settings, Experiment_settings)

# Training phase #############################################################################################
domain_settings['flipped_actions'] = False
performance = np.zeros((Experiment_settings['num_runs'], Experiment_settings['num_datapoints']))
test_agents = []
for run in range(Experiment_settings['num_runs']):
    if Experiment_settings['method'] == 1:
        test_agents.append(deepcopy(my_agent))
    if Experiment_settings['method'] == 2:
        summary_writer = update_summary_writer(agent_args, 'test', domain_settings)
        test_agent = MuZeroAgent(agent_config, domain_settings, summary_writer)
        test_agent.network = my_agent.network
        test_agents.append(test_agent)

    print(" ### run: ", run, " ############################")
    perf, eval_steps = test_agents[run].run_test(Experiment_settings)
    performance[run, :len(perf)] = perf[:]


print("time: {}s".format(time.time()-start))
save_results(Experiment_settings, filename, performance)
print('Done.')
