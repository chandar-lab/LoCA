################################################
#  Author: Harm van Seijen
#  Copyright 2020 Microsoft
################################################

import numpy as np

class ModelBased_agent(object):

    def __init__(self, settings, domain):
        self.domain = domain
        self.alpha = settings['alpha']
        self.epsilon_default = settings['epsilon']
        self.epsilon = self.epsilon_default
        self.transition_epsilon = settings['transition_epsilon']
        self.vi = settings['vi']

        self.iterations  = settings['model_iterations']
        self.r_init = settings['q_init']
        self.max_episode_length = settings['max_episode_length']

        self.eval_episodes = settings['eval_episodes']
        self.eval_epsilon = settings['eval_epsilon']
        self.eval_max_steps = settings['eval_max_steps']

        self.num_states = domain.get_num_states()
        self.num_actions = domain.get_num_actions()
        self.gamma = domain.get_gamma()
        self.trans_model = None
        self.trans_model_train = None
        self.trans_model_transition = None

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def reset_epsilon(self):
        self.epsilon = self.epsilon_default

    def _initialize(self):

        self.trans_model = np.zeros([self.num_states, self.num_actions, self.num_states])
        self.reward_model = np.ones([self.num_states, self.num_actions]) * self.r_init
        self.v = np.ones([self.num_states])*self.r_init

    def run_nopretrain(self, steps, num_datapoints):
        print("Run without pre-training.")
        self.domain.set_task(1)  # set domain to task A
        self.domain.set_phase('nopretrain')  # determines initial state distribution
        self._initialize()

        perf = self.run(steps, num_datapoints, eval=True, eval_type = 0)

        policy = self.get_current_policy()
        self.domain.show_policy(policy)

        print("avg reward: ", np.mean(perf), " final reward: ", perf[-1])
        return perf

    def run_train(self, steps):
        print("train phase started.")
        self.domain.set_task(0)  # set domain to task A
        self.domain.set_phase('train')
        if self.trans_model_train is None:
            self._initialize()
        else:
            self.trans_model = np.copy(self.trans_model_train)
            self.reward_model = np.copy(self.reward_model_train)
            self.v = np.copy(self.v_train)

        self.run(steps, 10)
        self.trans_model_train = np.copy(self.trans_model)
        self.reward_model_train = np.copy(self.reward_model)
        self.v_train = np.copy(self.v)
        self.behavior_policy = self.get_behavior_policy()

        policy = self.get_current_policy()
        self.domain.show_policy(policy)
        self.domain.show_value(self.v)

    def run_transition(self, steps):
        print("transition phase started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase('transition')

        assert self.trans_model_train is not None
        self.trans_model = np.copy(self.trans_model_train)
        self.reward_model = np.copy(self.reward_model_train)
        self.v = np.copy(self.v_train)

        self.run(steps, 10)
        self.trans_model_transition = np.copy(self.trans_model)
        self.reward_model_transition = np.copy(self.reward_model)
        self.v_transition = np.copy(self.v)

        policy = self.get_current_policy()
        self.domain.show_policy(policy)

    def run_test(self, steps, num_datapoints):
        print("test phase started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase('test')

        assert self.trans_model_transition is not None
        self.trans_model = np.copy(self.trans_model_transition)
        self.reward_model = np.copy(self.reward_model_transition)
        self.v = np.copy(self.v_transition)

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

        time = 0
        state = self.domain.get_initial_state()
        for i in range(steps):

            q = self.update_and_get_q(state)
            action = self.select_action(state,q)
            next_state, reward = self.domain.take_action(state, action); time += 1

            self.update_model(state,action,reward,next_state)

            if next_state == -1 or time == self.max_episode_length:
                time = 0
                state = self.domain.get_initial_state()
            else:
                state = next_state

            if eval:
                if ((i+1) % window_size == 0) & (data_point < num_datapoints):
                    performance[data_point] = self.eval_policy(eval_type)
                    data_point += 1

        return performance

    def update_model(self, state, action, reward, next_state):
        assert(state >= 0)
        next_state_vector = np.zeros([self.num_states])
        if next_state != -1:
            next_state_vector[next_state] = 1

        self.trans_model[state][action] += self.alpha*(next_state_vector - self.trans_model[state][action])
        self.reward_model[state][action] += self.alpha*(reward - self.reward_model[state][action])


    def update_and_get_q(self, state):
        if self.vi:
            self.iterate_over_model(self.iterations)
        else:
            self._update_current_state(state)
        q = self.compute_q_onestep_rollout(state)
        return q


    def _get_q(self):
        q = np.zeros([self.num_states, self.num_actions])
        for s in range(self.num_states):
            q[s] = self.compute_q_onestep_rollout(s)
        return q

    def _update_current_state(self, state):
        v_next = np.dot(self.trans_model[state], self.v)
        q = self.reward_model[state] + self.gamma * v_next
        self.v[state] = np.max(q)


    def iterate_over_model(self, iterations):
        for i in range(iterations):
            v_next = np.dot(self.trans_model, self.v)
            q = self.reward_model + self.gamma * v_next
            self.v = np.max(q, axis=1)

    def compute_q_onestep_rollout(self,state):
        v_next = np.dot(self.trans_model[state], self.v)
        q = self.reward_model[state] + self.gamma * v_next
        return q

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
        q = self._get_q()
        return self._get_egreedy_policy(q, self.eval_epsilon)

    def get_current_policy(self):
        q = self._get_q()
        return self._get_egreedy_policy(q, 0)

    def get_behavior_policy(self):
        q = self._get_q()
        return self._get_egreedy_policy(q, self.transition_epsilon)

    def _get_egreedy_policy(self, q, epsilon):
        policy = np.zeros([self.num_states, self.num_actions])
        for s in range(self.num_states):
            qmax = np.max(q[s])
            max_indices = np.nonzero(q[s] == qmax)[0]
            num_max = max_indices.size
            policy[s] = np.ones(self.num_actions) * epsilon / self.num_actions
            policy[s][max_indices] += (1 - epsilon) / num_max
        return policy

    def _get_egreedy_policy_local(self, q, epsilon):
        qmax = np.max(q)
        max_indices = np.nonzero(q == qmax)[0]
        num_max = max_indices.size
        local_policy = np.ones(self.num_actions) * epsilon / self.num_actions
        local_policy[max_indices] += (1 - epsilon) / num_max
        return local_policy



    def select_from_policy(self,state, policy):
        rnd = np.random.random()/1.0000001
        sum_p = 0.0
        for a in range(self.num_actions):
            sum_p += policy[state][a]
            if rnd < sum_p:
                return a
        assert False, "action not selected"


    def select_action(self,state, q):
        # selects e-greedy optimal action.
        if np.random.random() < self.epsilon:
            action= np.random.randint(self.num_actions)
        else:
            qmax = np.max(q)
            max_indices = np.nonzero(q == qmax)[0]
            num_max = max_indices.size
            i = np.random.randint(num_max)
            action = max_indices[i]

        return action


def test():
    q = np.array([3, 2, 3, 3])
    agent = ModelBased_agent(0,0)
    policy = agent.compute_local_policy(q)
    print(policy)




if __name__ == "__main__":
    test()