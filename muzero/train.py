import logging

import ray
import torch
import torch.optim as optim
from torch.nn import L1Loss

from .mcts import MCTS, Node
from .replay_buffer import ReplayBuffer
from .evaluate import evaluate
from .utils import select_action
import time

train_logger = logging.getLogger('train')
evaluate_logger = logging.getLogger('train_evaluate')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def _log(config, step_count, log_data, model, replay_buffer, lr, worker_logs, summary_writer):
    loss_data, td_data, priority_data = log_data
    weighted_loss, loss, policy_loss, reward_loss, value_loss = loss_data
    target_reward, target_value, trans_target_reward, trans_target_value, target_reward_phi, target_value_phi, \
    pred_reward, pred_value, target_policies, predicted_policies = td_data
    batch_weights, batch_indices = priority_data
    worker_reward, worker_disc_reward, worker_eps_len, evaluate_score, temperature, visit_entropy = worker_logs

    replay_episodes_collected = ray.get(replay_buffer.episodes_collected.remote())
    replay_buffer_size = ray.get(replay_buffer.size.remote())

    _msg = '#{:<10} Loss: {:<8.3f} [weighted Loss:{:<8.3f} Policy Loss: {:<8.3f} Value Loss: {:<8.3f} ' \
           'Reward Loss: {:<8.3f} ] Replay Episodes Collected: {:<10d} Buffer Size: {:<10d} Lr: {:<8.3f}'
    _msg = _msg.format(step_count, loss, weighted_loss, policy_loss, value_loss, reward_loss,
                       replay_episodes_collected, replay_buffer_size, lr)
    train_logger.info(_msg)

    if evaluate_score is not None:
        evaluate_msg = '#{:<10} evaluate Score: {:<10}'.format(step_count, evaluate_score)
        evaluate_logger.info(evaluate_msg)

    if summary_writer is not None:
        if config.debug:
            for name, W in model.named_parameters():
                summary_writer.add_histogram('after_grad_clip' + '/' + name + '_grad', W.grad.data.cpu().numpy(),
                                             step_count)
                summary_writer.add_histogram('network_weights' + '/' + name, W.data.cpu().numpy(), step_count)
            pass
        summary_writer.add_histogram('replay_data/replay_buffer_priorities',
                                     ray.get(replay_buffer.get_priorities.remote()),
                                     step_count)
        summary_writer.add_histogram('replay_data/batch_weight', batch_weights, step_count)
        summary_writer.add_histogram('replay_data/batch_indices', batch_indices, step_count)
        summary_writer.add_histogram('train_data_dist/target_reward', target_reward.flatten(), step_count)
        summary_writer.add_histogram('train_data_dist/target_value', target_value.flatten(), step_count)
        summary_writer.add_histogram('train_data_dist/transformed_target_reward', trans_target_reward.flatten(),
                                     step_count)
        summary_writer.add_histogram('train_data_dist/transformed_target_value', trans_target_value.flatten(),
                                     step_count)
        summary_writer.add_histogram('train_data_dist/target_reward_phi', target_reward_phi.unique().flatten(),
                                     step_count)
        summary_writer.add_histogram('train_data_dist/target_value_phi', target_value_phi.unique().flatten(),
                                     step_count)
        summary_writer.add_histogram('train_data_dist/pred_reward', pred_reward.flatten(), step_count)
        summary_writer.add_histogram('train_data_dist/pred_value', pred_value.flatten(), step_count)
        summary_writer.add_histogram('train_data_dist/pred_policies', predicted_policies.flatten(), step_count)
        summary_writer.add_histogram('train_data_dist/target_policies', target_policies.flatten(), step_count)

        summary_writer.add_scalar('train/loss', loss, step_count)
        summary_writer.add_scalar('train/weighted_loss', weighted_loss, step_count)
        summary_writer.add_scalar('train/policy_loss', policy_loss, step_count)
        summary_writer.add_scalar('train/value_loss', value_loss, step_count)
        summary_writer.add_scalar('train/reward_loss', reward_loss, step_count)
        summary_writer.add_scalar('train/episodes_collected', ray.get(replay_buffer.episodes_collected.remote()),
                                  step_count)
        summary_writer.add_scalar('train/replay_buffer_len', ray.get(replay_buffer.size.remote()), step_count)
        summary_writer.add_scalar('train/lr', lr, step_count)

        if worker_reward is not None:
            summary_writer.add_scalar('workers/reward', worker_reward, step_count)
            summary_writer.add_scalar('workers/disc_reward', worker_disc_reward, step_count)
            summary_writer.add_scalar('workers/eps_len', worker_eps_len, step_count)
            summary_writer.add_scalar('workers/temperature', temperature, step_count)
            summary_writer.add_scalar('workers/visit_entropy', visit_entropy, step_count)

        if evaluate_score is not None:
            summary_writer.add_scalar('train/evaluate_score', evaluate_score, step_count)


@ray.remote
class SharedStorage(object):
    def __init__(self, model):
        self.step_counter = 0
        self.model = model
        self.reward_log = []
        self.disc_reward_log = []
        self.evaluate_log = []
        self.evaluate_log_all, self.evaluate_steps = [], []
        self.eps_lengths = []
        self.temperature_log = []
        self.visit_entropies_log = []

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_data_worker_logs(self, eps_len, eps_reward, disc_reward, temperature, visit_entropy):
        self.eps_lengths.append(eps_len)
        self.reward_log.append(eps_reward)
        self.disc_reward_log.append(disc_reward)
        self.temperature_log.append(temperature)
        self.visit_entropies_log.append(visit_entropy)

    def add_evaluate_log(self, score):
        self.evaluate_log.append(score)
        self.evaluate_log_all.append(score)

    def add_evaluate_steps(self, steps):
        self.evaluate_steps.append(steps)

    def get_evaluate_log(self):
        return self.evaluate_log_all, self.evaluate_steps

    def get_worker_logs(self):
        if len(self.reward_log) > 0:
            reward = sum(self.reward_log) / len(self.reward_log)
            disc_reward = sum(self.disc_reward_log) / len(self.disc_reward_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)
            temperature = sum(self.temperature_log) / len(self.temperature_log)
            visit_entropy = sum(self.visit_entropies_log) / len(self.visit_entropies_log)

            self.reward_log = []
            self.eps_lengths = []
            self.disc_reward_log = []
            self.temperature_log = []
            self.visit_entropies_log = []

        else:
            reward = None
            disc_reward = None
            eps_lengths = None
            temperature = None
            visit_entropy = None

        if len(self.evaluate_log) > 0:
            evaluate_score = sum(self.evaluate_log) / len(self.evaluate_log)
            self.evaluate_log = []
        else:
            evaluate_score = None

        return reward, disc_reward, eps_lengths, evaluate_score, temperature, visit_entropy


@ray.remote
class DataWorker(object):
    def __init__(self, rank, config, domain_settings, shared_storage, replay_buffer):
        self.rank = rank
        self.config = config
        self.domain_settings = domain_settings
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

    def run(self):
        model = self.config.get_uniform_network()
        with torch.no_grad():
            while ray.get(self.shared_storage.get_counter.remote()) < self.config.training_steps and self.shared_storage is not None:
                model.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                model.eval()
                env = self.config.new_game(self.domain_settings, self.config.seed + self.rank)

                if env.phase == 'train':
                    env.set_task(0)
                    env.set_phase('train')
                elif env.phase == 'transition':
                    env.set_task(1)
                    env.set_phase('transition')
                elif env.phase == 'test':
                    env.set_task(1)
                    env.set_phase('test')

                print('Init state range: ', env.init_state)
                obs = env.reset()
                terminal = 0
                G, total_discount = 0, 1
                priorities = []
                eps_reward, eps_steps, visit_entropies = 0, 0, 0
                trained_steps = ray.get(self.shared_storage.get_counter.remote())
                _temperature = self.config.visit_softmax_temperature_fn(num_moves=len(env.history),
                                                                        trained_steps=trained_steps)
                while terminal == 0 and eps_steps <= self.config.max_moves:
                    root = Node(0)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                    network_output = model.initial_inference(obs)
                    root.expand(env.to_play(), env.legal_actions(), network_output)
                    root.add_exploration_noise(dirichlet_alpha=self.config.root_dirichlet_alpha,
                                               exploration_fraction=self.config.root_exploration_fraction)
                    MCTS(self.config).run(root, env.action_history(), model)
                    action, visit_entropy = select_action(root, temperature=_temperature, deterministic=False)
                    obs, reward, terminal, info = env.step(action.index)
                    env.store_search_statistics(root)

                    eps_reward += reward
                    G += total_discount * reward
                    total_discount *= env.gamma
                    eps_steps += 1
                    visit_entropies += visit_entropy

                    if not self.config.use_max_priority:
                        error = L1Loss(reduction='none')(network_output.value,
                                                         torch.tensor([[root.value()]])).item()
                        priorities.append(error + 1e-5)

                performance, _ = ray.get(self.shared_storage.get_evaluate_log.remote())

                if ray.get(self.shared_storage.get_counter.remote()) > 5000 and sum(performance[-100:]) / 100 > 0.98:
                    break

                print('Terminal Ended:{}, reward: {}, Init state: {}'.format(terminal, eps_reward, env.init))
                env.close()
                self.replay_buffer.save_game.remote(env,
                                                    priorities=None if self.config.use_max_priority else priorities)
                # Todo: refactor with env attributes to reduce variables
                visit_entropies /= eps_steps
                self.shared_storage.set_data_worker_logs.remote(eps_steps, eps_reward, G, _temperature, visit_entropies)


def update_weights(model, target_model, optimizer, replay_buffer, config):
    batch = ray.get(replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps,
                                                      model=target_model if config.use_target_model else None,
                                                      config=config))
    obs_batch, action_batch, target_reward, target_value, target_policy, indices, weights = batch

    obs_batch = obs_batch.to(config.device)
    action_batch = action_batch.to(config.device).unsqueeze(-1)
    target_reward = target_reward.to(config.device)
    target_value = target_value.to(config.device)
    target_policy = target_policy.to(config.device)
    weights = weights.to(config.device)

    # transform targets to categorical representation
    # Reference:  Appendix F
    transformed_target_reward = config.scalar_transform(target_reward)
    target_reward_phi = config.reward_phi(transformed_target_reward)
    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)
    value, _, policy_logits, hidden_state = model.initial_inference(obs_batch)

    scaled_value = config.inverse_value_transform(value)
    # Note: Following line is just for logging.
    predicted_values, predicted_rewards, predicted_policies = scaled_value, None, torch.softmax(policy_logits, dim=1)

    # Reference: Appendix G
    new_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
    new_priority += 1e-5
    new_priority = new_priority.data.cpu().numpy()

    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    reward_loss = torch.zeros(config.batch_size, device=config.device)

    gradient_scale = 1 / config.num_unroll_steps
    for step_i in range(config.num_unroll_steps):
        value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_batch[:, step_i])
        policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
        value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
        reward_loss += config.scalar_reward_loss(reward, target_reward_phi[:, step_i])
        hidden_state.register_hook(lambda grad: grad * 0.5)

        # collected for logging
        predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value)))
        scaled_rewards = config.inverse_reward_transform(reward)
        predicted_rewards = scaled_rewards if predicted_rewards is None else torch.cat((predicted_rewards,
                                                                                        scaled_rewards))
        predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1)))

    # optimize
    loss = (policy_loss + config.value_loss_coeff * value_loss + reward_loss)
    weighted_loss = (weights * loss).mean()
    weighted_loss.register_hook(lambda grad: grad * gradient_scale)
    loss = loss.mean()

    optimizer.zero_grad()
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    # update priorities
    replay_buffer.update_priorities.remote(indices, new_priority)

    # packing data for logging
    loss_data = (weighted_loss.item(), loss.item(), policy_loss.mean().item(), reward_loss.mean().item(),
                 value_loss.mean().item())
    td_data = (target_reward, target_value, transformed_target_reward, transformed_target_value,
               target_reward_phi, target_value_phi, predicted_rewards, predicted_values,
               target_policy, predicted_policies)
    priority_data = (weights, indices)

    return loss_data, td_data, priority_data


def adjust_lr(config, optimizer, step_count):
    lr = config.lr_init * config.lr_decay_rate ** (step_count / config.lr_decay_steps)
    lr = max(lr, 0.001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def _train(config, shared_storage, replay_buffer, domain_settings, summary_writer):
    model = config.get_uniform_network().to(config.device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    target_model = config.get_uniform_network().to('cpu')
    target_model.eval()

    # wait for replay buffer to be non-empty
    while ray.get(replay_buffer.size.remote()) == 0:
        pass

    for step_count in range(config.training_steps):
        shared_storage.incr_counter.remote()
        lr = adjust_lr(config, optimizer, step_count)

        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(model.get_weights())

        log_data = update_weights(model, target_model, optimizer, replay_buffer, config)

        # softly update target model
        if config.use_target_model:
            soft_update(target_model, model, tau=1e-2)
            target_model.eval()

        if ray.get(shared_storage.get_counter.remote()) % 50 == 0:
            _log(config, step_count, log_data, model, replay_buffer, lr,
                 ray.get(shared_storage.get_worker_logs.remote()), summary_writer)

            replay_buffer.remove_to_fit.remote()
        performance, _ = ray.get(shared_storage.get_evaluate_log.remote())

        if step_count > 5000 and sum(performance[-100:])/100 > 0.98:
            shared_storage.step_counter = config.training_steps
            break

    shared_storage.set_weights.remote(model.get_weights())


@ray.remote
def _evaluate(config, domain_settings, shared_storage):
    evaluate_model = config.get_uniform_network().to('cpu')
    best_evaluate_score = float('-inf')

    while ray.get(shared_storage.get_counter.remote()) < config.training_steps:
        if ray.get(shared_storage.get_counter.remote()) > 0 and \
                ray.get(shared_storage.get_counter.remote()) % config.evaluate_interval == 0:
            print('Evaluation started at episode: {}'.format(ray.get(shared_storage.get_counter.remote())))
            shared_storage.add_evaluate_steps.remote(ray.get(shared_storage.get_counter.remote()))
            evaluate_model.set_weights(ray.get(shared_storage.get_weights.remote()))
            evaluate_model.eval()

            evaluate_score = evaluate(config, domain_settings, evaluate_model, config.evaluate_episodes, 'cpu', False)
            if evaluate_score >= best_evaluate_score:
                best_evaluate_score = evaluate_score
                print('best_evaluate_score: {}'.format(best_evaluate_score))
                torch.save(evaluate_model.state_dict(), config.model_path)
            if domain_settings['phase'] == 'transition':
                # It always saves the model in transition phase regardless of the performance
                torch.save(evaluate_model.state_dict(), config.model_path)

            shared_storage.add_evaluate_log.remote(evaluate_score)
            performance, _ = ray.get(shared_storage.get_evaluate_log.remote())
            time.sleep(10)
            if ray.get(shared_storage.get_counter.remote()) > 5000 and sum(performance[-100:]) / 100 > 0.98:
                break


def train(config, domain_settings, network, summary_writer=None):
    storage = SharedStorage.remote(network)
    replay_buffer = ReplayBuffer.remote(batch_size=config.batch_size, capacity=config.window_size,
                                        prob_alpha=config.priority_prob_alpha)
    workers = [DataWorker.remote(rank, config, domain_settings, storage, replay_buffer).run.remote()
               for rank in range(0, config.num_actors)]

    workers += [_evaluate.remote(config, domain_settings, storage)]
    _train(config, storage, replay_buffer, domain_settings, summary_writer)
    ray.wait(workers, len(workers))
    network.set_weights(ray.get(storage.get_weights.remote()))

    return network


def test(config, domain_settings, trained_net, summary_writer=None):
    storage = SharedStorage.remote(trained_net)
    replay_buffer = ReplayBuffer.remote(batch_size=config.batch_size, capacity=config.window_size,
                                        prob_alpha=config.priority_prob_alpha)
    workers = [DataWorker.remote(rank, config, domain_settings, storage, replay_buffer).run.remote()
               for rank in range(0, config.num_actors)]
    workers += [_evaluate.remote(config, domain_settings, storage)]
    _train(config, storage, replay_buffer, domain_settings, summary_writer)
    ray.wait(workers, len(workers))
    performance = ray.get(storage.get_evaluate_log.remote())
    print('Training performance: {}'.format(performance))
    trained_net = config.get_uniform_network().set_weights(ray.get(storage.get_weights.remote())),

    return trained_net, performance
