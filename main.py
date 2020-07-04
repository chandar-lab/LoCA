import numpy as np
import time

from utils import create_filename, save_results, update_summary_writer
from config import get_domain_setting, get_experiment_setting

import argparse
import pickle
from copy import deepcopy

# Settings  ##########################################################################
parser = argparse.ArgumentParser(description='LoCA regret Experiments')
parser.add_argument('--method', default='sarsa_lambda', help='Name of the method')
parser.add_argument('--env', default='MountainCar', help='Name of the environment')
parser.add_argument('--no_pre_training',  action='store_true', default=False)
parser.add_argument('--load',  action='store_true', default=False, help='load previous pre-trained agents')
parser.add_argument('--save',  action='store_true', default=True, help='save the agent and the results')
parser.add_argument('--flipped_terminals',  action='store_true', default=False, help='flip the rewards associated '
                                                                                     'with terminal 1 and terminal 2')
parser.add_argument('--flipped_actions',  action='store_true', default=False, help='Shuffle the actions to cancel '
                                                                                   'the effect of model learning')

args = parser.parse_args()
experiment_settings = get_experiment_setting(args)
domain_settings = get_domain_setting(args)
filename = create_filename(args)
print("file: ", filename)
experiment_settings['filename'] = filename

if experiment_settings['method'] == 'sarsa_lambda':
    agent_config = []
    from sarsa_lambda.sarsa_lambda import build_agent, load_agent
elif experiment_settings['method'] == 'MuZero':
    from muzero.MuZeroAgent import MuZeroAgent, build_agent
    from muzero.env import muzero_config
    agent_config = muzero_config
    agent_config.flippedTask = args.flipped_terminals
    agent_config.flippedActions = args.flipped_actions
    from muzero.MuZeroAgent import build_agent, load_agent
else:
    assert False, 'HvS: Invalid method id.'
# Pre-Training phase ###########################################################################################
start = time.time()
my_agent = None
if not args.load:  # pre-train
    if args.no_pre_training is False:
        my_agent = build_agent(domain_settings, agent_config)
        my_agent.run_pretrain(experiment_settings, include_transition=not domain_settings['flipped_actions'])
    else:  # no pre-training
        agent_config.opr = 'test'
        my_agent = build_agent(domain_settings, agent_config)
    print("time: {}s".format(time.time()-start))

else:  # load an agent
    my_agent, agent_config = load_agent(agent_config, domain_settings, experiment_settings)
# Training phase #############################################################################################
experiment_settings['num_runs'] = 20
experiment_settings['num_test_steps'] = 30000
experiment_settings['num_datapoints'] = 100

domain_settings['flipped_actions'] = False
performance = np.zeros((experiment_settings['num_runs'], 2, experiment_settings['num_datapoints']))
test_agents = []
count = 0
my_agent.epsilon_init_test = 0.1
for run in range(experiment_settings['num_runs']):
    if experiment_settings['method'] == 'sarsa_lambda':
        test_agents.append(deepcopy(my_agent))
    if experiment_settings['method'] == 'MuZero':
        summary_writer = update_summary_writer(agent_config, 'test' + str(run), domain_settings)
        test_agent = MuZeroAgent(agent_config, domain_settings, summary_writer)
        test_agent.network = my_agent.network
        test_agents.append(test_agent)

    print(" ### run: ", run, " ############################")
    perf = test_agents[run].run_train(experiment_settings)
    if np.mean(perf[0][-10:]) > 0.8:
        performance[count, 0, :] = perf[0][:experiment_settings['num_datapoints']]
        performance[count, 1, :] = perf[1][:experiment_settings['num_datapoints']]
        count += 1
        print(count)

experiment_settings['num_runs'] = count
print("time: {}s".format(time.time()-start))
save_results(experiment_settings, filename, performance[:count, :, :])
print('Done.')
