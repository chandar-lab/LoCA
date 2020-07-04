################################################
#  Author: Harm van Seijen
#  Copyright 2020 Microsoft
################################################


import numpy as np
import time
import json
import math
from domain import Domain
from modelbased_nstep import ModelBased_nstep_agent
from modelbased import ModelBased_agent
from sarsa_lambda import SarsaLambda_agent
from qlearning import Qlearning_agent


# Settings  ######################################################################


# METHODS:
# 1:  mb_vi            - model-based method, value-iteration
# 2:  mb_su            - model-based method, single update (only current state)
# 3:  mb_nstep         - n-step model-based method
# 4:  sarsa_lambda
# 5:  qlearning


################################ MAIN PARAMETERS #################################

method = 4                 # see methods above
LoCA_pretraining = False    # with or without LoCA pretraining
alpha_multp = 1.0          # any value > 0
S_multp = 1                # {1, 2, 3, ... }
n_step = 1                 # only relevant when method =  3 (mb_nstep)

##################################################################################

num_train_steps =  1000000
num_transition_steps = 50000
num_test_steps = 100000
num_steps_nopretrain =  100000
num_datapoints = 200
num_runs = 10
filename_extension = ''

# Domain settings
domain_settings = {}
domain_settings['height'] = 4
domain_settings['width'] = 25
domain_settings['stochasticity'] = 0
domain_settings['gamma'] = 0.97
domain_settings['init_state_train'] = -1
domain_settings['init_state_transition'] = np.array([24])
domain_settings['init_state_test'] = -1
domain_settings['init_state_eval'] = np.arange(10,15)
domain_settings['init_state_nopretrain'] = -1
domain_settings['state_space_multiplier'] = S_multp

# Agent settings
agent_settings = {}
if method == 3:
    agent_settings['alpha'] = 0.04  * alpha_multp    # multi-step model requires lower step-size
elif method == 4:
    agent_settings['alpha'] = 0.05 * alpha_multp     # traces requires lower step-size
else:
    agent_settings['alpha'] = 0.2 * alpha_multp
agent_settings['lambda'] = 0.95  # only relevant for sarsa_lambda
agent_settings['nstep_model'] = n_step      # only relevant for modelbased_nstep
agent_settings['max_episode_length'] = 100
agent_settings['epsilon'] = 0.1
agent_settings['transition_epsilon'] = 0.1
agent_settings['q_init'] = 4
agent_settings['model_iterations'] = 5  # only relevant for mb_vi and mb_vi_onpolicy
agent_settings['eval_episodes'] = 10
agent_settings['eval_epsilon'] = 0
agent_settings['eval_max_steps'] = 40

#############################################################################################


if method == 1:
    method_name = 'mb_vi'
    agent_settings['vi'] = True
elif method == 2:
    method_name = 'mb_su'
    agent_settings['vi'] = False
elif method == 3:
    method_name = 'mb_nstep'
elif method == 4:
    method_name = 'sarsa_lambda'
elif method == 5:
    method_name = 'qlearning'
else:
    assert False, 'HvS: Invalid method id.'


#method_name = 'qlearning'
if LoCA_pretraining:
    filename = method_name + '_LoCA' + filename_extension
else:
    filename = method_name + '_noLoCA' + filename_extension


print("file: ", filename)


if num_datapoints > num_test_steps:
    num_datapoints = num_test_steps

my_domain = Domain(domain_settings)
if method in [1,2]:
    my_agent = ModelBased_agent(agent_settings, my_domain)
elif method == 3:
    my_agent = ModelBased_nstep_agent(agent_settings, my_domain)
elif method == 4:
    my_agent = SarsaLambda_agent(agent_settings, my_domain)
elif method == 5:
    my_agent = Qlearning_agent(agent_settings, my_domain)
else:
    assert False, "invalid method"


# Training phase
performance = np.zeros([num_runs, num_datapoints])
start = time.time()
if LoCA_pretraining:
    my_agent.run_train(num_train_steps)
    print("time: {}s".format(time.time()-start))

    for run in range(num_runs):
        print(''); print("### run: ", run, " ############################")
        my_agent.run_transition(num_transition_steps)
        performance[run, :] = my_agent.run_test(num_test_steps, num_datapoints)
else:
    for run in range(num_runs):
        print('');
        print("### run: ", run, " ############################")
        performance[run, :] = my_agent.run_nopretrain(num_steps_nopretrain, num_datapoints)

end = time.time()

avg_performance = np.mean(performance,axis=0)
print("time: {}s".format(end-start))
print("avg reward: ", np.mean(avg_performance), " final reward: ", avg_performance[-1])


# compute regret
if LoCA_pretraining:
    num_steps = num_test_steps
else:
    num_steps = num_steps_nopretrain
window_size = num_steps // num_datapoints
regret = np.zeros(num_runs)
for run in range(num_runs):
    regret[run] = num_datapoints*window_size - np.sum(performance[run])*window_size

# compute average and standard error
avg_regret = np.mean(regret)
std_error = 0
for run in range(num_runs):
    avg_factor = 1 / float(run + 1)
    std_error = (1 - avg_factor) * std_error + avg_factor * (regret[run] - avg_regret) ** 2
std_error = math.sqrt(std_error / float(num_runs))

print("REGRET (x1000) : {:3.2f}".format(avg_regret/1000), ", std error: {:3.2f}".format(std_error/1000))



# Store results + some essential settings
settings = {}
settings['method_name'] = method_name
if method == 4:
    settings['n_step'] = n_step
settings['avg_regret'] = avg_regret
settings['std_error'] = std_error
settings['num_steps'] = num_steps
settings['num_datapoints'] = num_datapoints
settings['num_runs'] = num_runs
settings['stochasticity'] = domain_settings['stochasticity']
settings['state_space_multiplier'] = domain_settings['state_space_multiplier']
settings['alpha'] = agent_settings['alpha']
settings['lambda']  = agent_settings['lambda']
settings['q_init'] = agent_settings['q_init']

print("file: ", filename)

with open('data/' + filename + '_settings.txt', 'w') as json_file:
    json.dump(settings, json_file)
np.save('data/' +  filename + '_results.npy', performance)


print('Done.')