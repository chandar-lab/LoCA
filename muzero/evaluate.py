import os
import torch

from .mcts import MCTS, Node
from .utils import select_action
import multiprocessing


def _evaluate(config, domain_setting, model, ep_i, device, render, save_video, save_path, ep_data):
    with torch.no_grad():
        env = config.new_game(domain_setting, save_video=save_video, save_path=save_path,
                              video_callable=lambda episode_id: True, uid=ep_i)
        if domain_setting['phase'] == 'train':
            env.set_task(0)
        else:
            env.set_task(1)
        env.set_eval_mode(True)
        terminal = 0
        ep_reward = 0
        steps = 0
        obs = env.reset()

        # print('####Evaluation: Init state range: ', env.init)
        while terminal == 0 and steps < config.max_moves:
            if render:
                env.render()
            root = Node(0)
            obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
            root.expand(env.to_play(), env.legal_actions(), model.initial_inference(obs))
            MCTS(config).run(root, env.action_history(), model)
            action, _ = select_action(root, temperature=1, deterministic=True)
            obs, reward, terminal, info = env.step(action.index)
            steps += 1
            if terminal != 0:
                # print('#### Evaluation ended at terminal: ', terminal)
                ep_reward += reward
                break
        env.set_eval_mode(False)

        env.close()

    ep_data[ep_i] = ep_reward


def evaluate(config, domain_settings, model, episodes, device, render, save_video=False):
    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings')

    manager = multiprocessing.Manager()
    ep_data = manager.dict()
    jobs = []
    for ep_i in range(episodes):
        p = multiprocessing.Process(target=_evaluate, args=(config, domain_settings, model, ep_i, device, render, save_video, save_path,
                                                        ep_data))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    evaluate_reward = sum(ep_data.values())
    print('######### Evaluation Done!, {} '.format(evaluate_reward / episodes))

    return evaluate_reward / episodes
