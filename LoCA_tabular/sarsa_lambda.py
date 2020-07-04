################################################
#  Author: Harm van Seijen
#  Copyright 2020 Microsoft
################################################

import numpy as np

class SarsaLambda_agent(object):

    def __init__(self, settings, domain):

        self.domain = domain
        self.alpha = settings['alpha']
        self.lAmbda = settings['lambda']
        self.type = 2   # traces type  --- 0: accumulating traces, 1; replacing traces, 2: dutch traces
        self.epsilon_default = settings['epsilon']
        self.epsilon = self.epsilon_default
        self.transition_epsilon = settings['transition_epsilon']
        self.q_init = settings['q_init']
        self.eval_episodes = settings['eval_episodes']
        self.eval_epsilon = settings['eval_epsilon']
        self.eval_max_steps = settings['eval_max_steps']
        self.max_episode_length = settings['max_episode_length']

        self.num_states = domain.get_num_states()
        self.num_actions = domain.get_num_actions()
        self.gamma = domain.get_gamma()
        self.q = None
        self.q_train = None
        self.q_transition = None


    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def reset_epsilon(self):
        self.epsilon = self.epsilon_default

    def run_nopretrain(self, steps, num_datapoints):
        print("Run without pre-training.")
        self.domain.set_task(1)  # set domain to task A
        self.domain.set_phase('nopretrain')  # determines initial state distribution
        self.q = np.ones([self.num_states, self.num_actions]) * self.q_init

        perf = self.run(steps, num_datapoints, eval=True, eval_type = 0)

        policy = self.get_current_policy()
        self.domain.show_policy(policy)

        print("avg reward: ", np.mean(perf), " final reward: ", perf[-1])
        return perf

    def run_train(self, steps, flipped=False):
        print("train phase started.")
        if flipped:
            self.domain.set_task(2)  # set domain to task A
        else:
            self.domain.set_task(0)  # set domain to task A
        self.domain.set_phase('train')
        if self.q_train is None:
            self.q = np.ones([self.num_states, self.num_actions]) * self.q_init
        else:
            self.q = np.copy(self.q_train)

        self.run(steps, 10)
        self.q_train = np.copy(self.q)
        self.behavior_policy = self.get_behavior_policy()

        policy = self.get_current_policy()
        self.domain.show_policy(policy)
        v = np.sum(np.multiply(policy, self.q), axis=1)
        self.domain.show_value(v)


    def run_transition(self, steps):
        print("transition phase started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase('transition')
        assert self.q_train is not None
        self.q = np.copy(self.q_train)

        self.run(steps, 10)
        self.q_transition = np.copy(self.q)

        policy = self.get_current_policy()
        self.domain.show_policy(policy)


    def run_test(self, steps, num_datapoints):
        print("test phase started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase('test')
        assert self.q_transition is not None
        self.q = np.copy(self.q_transition)

        perf = self.run(steps, num_datapoints, eval=True)

        policy = self.get_current_policy()
        self.domain.show_policy(policy)

        print("avg reward: ", np.mean(perf), " final reward: ", perf[-1])
        return perf


    def run(self, steps, num_datapoints, eval=False, eval_type = 0):
        # perform a run over certain number of time steps

        window_size = steps // num_datapoints
        data_point = 0

        performance = np.zeros([num_datapoints])

        next_state = -1
        for i in range(steps):

            if next_state == -1 or time == self.max_episode_length:
                time = 0
                state = self.domain.get_initial_state()
                self._reset_trace()
                action = self.select_action(state)
            else:
                state = next_state
                action = next_action

            next_state, reward = self.domain.take_action(state, action);  time += 1
            next_action = self.select_action(next_state)


            # perform update
            self._perform_update(state, action, reward, next_state, next_action)


            if eval:
                if ((i+1) % window_size == 0) & (data_point < num_datapoints):
                    performance[data_point] = self.eval_policy(eval_type)
                    data_point += 1

        return performance

    def _reset_trace(self):
        self.e_trace = np.zeros([self.num_states, self.num_actions])
        self.q_sa_old = 0

    def _perform_update(self, state, action, reward, next_state, next_action):

        q_sa = self.q[state][action]
        if next_state == -1:
            q_sa_next = 0
        else:
            q_sa_next = self.q[next_state][next_action]

        if self.type == 2:
            delta = reward + self.gamma * q_sa_next - self.q_sa_old
        else:
            delta = reward + self.gamma * q_sa_next - q_sa


        # update traces
        if self.type == 0:  # accumulating traces
            self.e_trace[state][action] += self.alpha
        elif self.type == 1:  # replacing traces
            self.e_trace[state][action] = self.alpha
        elif self.type == 2:  # dutch traces
            e_phi = self.e_trace[state][action]
            self.e_trace[state][action] += self.alpha * (1 - e_phi)
        else:
            assert False

        self.q += self.e_trace * delta

        if self.type == 2:
            self.q[state][action] -= self.alpha * (q_sa - self.q_sa_old)
            self.q_sa_old = q_sa_next

        self.e_trace *= self.gamma * self.lAmbda


    def _eval_policy_returns(self, policy):
        sum_returns = 0
        for ep in range(self.eval_episodes):
            R = 0
            discount_factor = 1
            state = self.domain.get_initial_state()
            for i in range(self.eval_max_steps):
                action = self.select_from_policy(state, policy)
                state, reward = self.domain.take_action(state, action)
                R += discount_factor * reward
                discount_factor *= self.gamma
                if state == -1:
                    sum_returns += R
                    break
        return sum_returns / self.eval_episodes


    def _eval_policy_default(self, policy):
        self.domain.set_eval_mode(True)
        sum_rewards = 0
        for ep in range(self.eval_episodes):
            state = self.domain.get_initial_state()
            for i in range(self.eval_max_steps):
                action = self.select_from_policy(state, policy)
                state, reward = self.domain.take_action(state, action)
                if state == -1:
                    sum_rewards += reward
                    break
        self.domain.set_eval_mode(False)
        return sum_rewards / self.eval_episodes


    def eval_policy(self, eval_type):
        policy = self.get_eval_policy()
        if eval_type == 0:
            performance = self._eval_policy_default(policy)
        elif eval_type == 1:
            performance = self._eval_policy_returns(policy)
        else:
            assert False, "invalid evaluation-type"
        return performance


    def get_eval_policy(self):
        return self._get_egreedy_policy(self.q, self.eval_epsilon)

    def get_current_policy(self):
        return self._get_egreedy_policy(self.q, self.epsilon)

    def get_behavior_policy(self):
        return self._get_egreedy_policy(self.q, self.transition_epsilon)

    def _get_egreedy_policy(self, q, epsilon):
        policy = np.zeros([self.num_states, self.num_actions])
        for s in range(self.num_states):
            qmax = np.max(q[s])
            max_indices = np.nonzero(q[s] == qmax)[0]
            num_max = max_indices.size
            policy[s] = np.ones(self.num_actions) * epsilon / self.num_actions
            policy[s][max_indices] += (1 - epsilon) / num_max
        return policy


    def select_from_policy(self,state, policy):
        rnd = np.random.random()/1.0000001
        sum_p = 0.0
        for a in range(self.num_actions):
            sum_p += policy[state][a]
            if rnd < sum_p:
                return a
        assert False, "action not selected"


    def select_action(self,state):
        # selects e-greedy optimal action.
        if np.random.random() < self.epsilon:
            action= np.random.randint(self.num_actions)
        else:
            qmax = np.max(self.q[state])
            max_indices = np.nonzero(self.q[state] == qmax)[0]
            num_max = max_indices.size
            i = np.random.randint(num_max)
            action = max_indices[i]

        return action