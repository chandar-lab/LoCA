import gym
import torch
from ..config import BaseMuZeroConfig, DiscreteSupport
from .env_wrapper import ClassicControlWrapper
from .model import MuZeroNet
from .mountain_car import MountainCarEnv


class ClassicControlConfig(BaseMuZeroConfig):
    def __init__(self):
        super(ClassicControlConfig, self).__init__(
            training_steps=20000,
            evaluate_interval=1000,
            evaluate_episodes=3,
            checkpoint_interval=20,
            max_moves=300,
            discount=0.997,
            dirichlet_alpha=0.35,
            num_simulations=50,
            batch_size=64,
            td_steps=3,
            num_actors=2,
            lr_init=0.1,
            lr_decay_rate=0.01,
            lr_decay_steps=200000,
            window_size=10000,
            value_loss_coeff=10,
            value_support=DiscreteSupport(-1, 4),
            reward_support=DiscreteSupport(-1, 4))

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def set_game(self, env, setting, save_video=False, save_path=None, video_callable=None):
        self.env = env
        game = self.new_game(setting)
        self.obs_shape = game.reset().shape[0]
        self.action_space_size = game.action_space_size
        return game

    def get_uniform_network(self):
        return MuZeroNet(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
                         self.inverse_value_transform, self.inverse_reward_transform)

    def new_game(self, setting, seed=None, save_video=False, save_path=None, video_callable=None, uid=None):
        # env = gym.make(self.env_name)

        env = MountainCarEnv(setting)

        if seed is not None:
            env.seed(seed)

        return env
        # return ClassicControlWrapper(env, discount=setting['gamma'], k=4)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


muzero_config = ClassicControlConfig()
