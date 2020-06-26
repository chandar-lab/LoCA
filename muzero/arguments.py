import os
import torch

class get_muzero_args(object):
    def __init__(self):
        self.env = 'MountainCar' # 'Name of the environment'
        self.result_dir = os.path.join(os.getcwd()) + '/results/MuZero'
        self.opr = 'train' # choices=['train', 'test']
        self.no_cuda = False
        self.debug = False
        self.render = False
        self.force = True # 'Overrides past results (default: %(default)s)'
        self.seed = 0
        self.value_loss_coeff = None # 'scale for value loss (default: %(default)s)
        self.revisit_policy_search_rate = None # 'Rate at which target policy is re-estimated (default: %(default)s)'
        self.use_max_priority = False # 'Forces max priority assignment for new incoming data in replay buffer '
        self.use_priority = False # 'Uses priority for data sampling in replay buffer. '
                             # 'Also, priority for new data is calculated based on loss (default: False)'
        self.use_target_model = False # 'Use target model for bootstrap value estimation (default: %(default)s)'
        self.test_episodes = 10 # 'Evaluation episode count (default: %(default)s)'
        self.flippedTask = False # Flipped terminal in the case of tow terminals
        self.flippedActions = False # Shuffling the actions to cancel the effect of model learning


        self.device = 'cuda' if (not self.no_cuda) and torch.cuda.is_available() else 'cpu'
        assert self.revisit_policy_search_rate is None or 0 <= self.revisit_policy_search_rate <= 1, \
            ' Revisit policy search rate should be in [0,1]'
