

def get_sarsa_settings():
    """
        This function returns sarsa_lambda settings
        Args:

        Returns:
            sarsa_lambda settings
        """
    agent_settings = {}
    agent_settings['alpha'] = 0.5 / 10
    agent_settings['lambda'] = 0.9  # only relevant for sarsa_lambda
    agent_settings[
        'traces_type'] = 2  # only relevant for sarsa_lambda  -- 0: accumulating traces, 1; replacing traces, 2: dutch traces
    agent_settings['epsilon_init'] = 1
    agent_settings['epsilon_init_test'] = 0.15
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
