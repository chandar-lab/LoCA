import os


def get_doamin_setting(args):
    """
        This function returns domain settings
        Args:
            This function only needs args.flipped_terminals, args.flipped_actions
            inital state distributions can be chosen based on the domain
        Returns:
            domain_settings
        """

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
    """
        Methods implemented: 1- sarsa_lambda, 2- MuZero
        Args:
            This function only needs args.env, args.method and args.save
            The numeber of training steps and test steps should be adopted for each domain
        Returns:
            experiment_settings
        """

    experiment_settings = {}
    experiment_settings['env'] = args.env
    experiment_settings['method'] = args.method
    # experiment_settings['num_train_steps1'] = 100000
    experiment_settings['num_train_steps'] = 5000000
    experiment_settings['num_transition_steps'] = 5000 #5000
    experiment_settings['num_test_steps'] = 20000
    experiment_settings['num_datapoints'] = 50
    experiment_settings['num_runs'] = 20
    experiment_settings['save'] = args.save
    experiment_settings['filename'] = None

    if experiment_settings['num_datapoints'] > experiment_settings['num_test_steps']:
        experiment_settings['num_datapoints'] = experiment_settings['num_test_steps']

    return experiment_settings
