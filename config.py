import os


def get_sarsa_setting():
    agent_settings = {}
    agent_settings['alpha'] = 0.5 / 10
    agent_settings['lambda'] = 0.9  # only relevant for sarsa_lambda
    agent_settings[
        'traces_type'] = 2  # only relevant for sarsa_lambda  -- 0: accumulating traces, 1; replacing traces, 2: dutch traces
    agent_settings['epsilon_init'] = 1
    agent_settings['epsilon_init_test'] = 0.1
    agent_settings['ep_decay_rate'] = 0.1
    agent_settings['epsilon_decay_steps_train'] = 1000
    agent_settings['epsilon_decay_steps_test'] = 1000
    agent_settings['transition_epsilon'] = 0.1
    agent_settings['theta_init'] = 0
    agent_settings['search_horizon'] = 3  # only relevant for mb_mpc and mb_mpc_onpolicy
    agent_settings['model_iterations'] = 10  # only relevant for mb_vi and mb_vi_onpolicy
    agent_settings['eval_episodes'] = 20
    agent_settings['eval_epsilon'] = 0
    agent_settings['train_max_steps'] = 10000
    agent_settings['eval_max_steps'] = 400
    agent_settings['fixed_behavior'] = False
    agent_settings['type'] = 2  # only relevant for Sarsa_lambda. 0: accumulating traces, 1: replacing traces, 2: true online

    return agent_settings


def get_doamin_setting(args):

    domain_settings = {}
    domain_settings['flipped_terminals'] = args.flipped_terminals
    domain_settings['flipped_actions'] = args.flipped_actions
    # Flipped action experiment is done only in training phase
    if args.no_pre_training or args.load:
        domain_settings['flipped_actions'] = False

    if not domain_settings['flipped_terminals']:
        domain_settings['init_state_transition'] = [[[0.4, 0.5], [0, 0.07]]]
    else:
        domain_settings['init_state_transition'] = [[[-0.61, -0.44], [-0.008, 0.008]]]

    domain_settings['num_tilings'] = 10
    domain_settings['num_x_tiles'] = 10
    domain_settings['num_v_tiles'] = 10
    domain_settings['gamma'] = 0.997
    domain_settings['init_state_train'] = [[[-1.2, 0.5], [-0.07, 0.07]], [[-1, 0], [-0.03, 0.03]]]
    domain_settings['init_state_test'] = domain_settings['init_state_train']
    domain_settings['init_state_eval'] = [[[-0.2, -0.1], [-0.01, 0.01]]]  # to find
    domain_settings['reward_taskB_R'] = 1
    domain_settings['max_steps_per_episode'] = 400
    domain_settings['phase'] = 'train'
    return domain_settings


def get_experiment_setting(args):
    # Experiment settings
    # METHODS:
    # 1:  sarsa_lambda
    # 2:  MuZero

    experiment_settings = {}
    experiment_settings['env'] = args.env
    experiment_settings['method'] = args.method
    experiment_settings['num_train_steps1'] = 100000
    experiment_settings['num_train_steps'] = 9000 #1000000
    experiment_settings['num_transition_steps'] = 500 #5000
    experiment_settings['num_test_steps'] = 9000 #30000
    experiment_settings['num_datapoints'] = 3
    experiment_settings['num_runs'] = 3
    experiment_settings['save'] = args.save
    experiment_settings['filename'] = None

    if experiment_settings['num_datapoints'] > experiment_settings['num_test_steps']:
        experiment_settings['num_datapoints'] = experiment_settings['num_test_steps']

    return experiment_settings
