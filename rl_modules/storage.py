import numpy as np
import torch


class Storage:
    class Transition:  # Represents a single transition in the environment,
        def __init__(self):
            self.obs = None
            self.action = None
            self.reward = None
            self.done = None
            self.value = None
            self.action_log_prob = None

        # clear method resets the transition to its initial state.
        def clear(self):
            self.__init__()

    def __init__(self,
                 obs_dim,
                 action_dim,
                 max_timesteps,  # Maximum number of timesteps to store.
                 gamma=0.8):  # Discount factor for computing returns.
        self.max_timesteps = max_timesteps
        self.gamma = gamma

        # create the buffer
        self.obs = np.zeros([self.max_timesteps, obs_dim])
        self.actions = np.zeros([self.max_timesteps, action_dim])
        self.rewards = np.zeros([self.max_timesteps])
        self.dones = np.zeros([self.max_timesteps])
        # For RL methods
        self.actions_log_prob = np.zeros([self.max_timesteps])
        self.values = np.zeros([self.max_timesteps])
        self.returns = np.zeros([self.max_timesteps])
        self.advantages = np.zeros([self.max_timesteps])

        self.step = 0
        self.lambda_ = 0.99

    # store transition into buffer, and increase step number
    def store_transition(self, transition: Transition):
        # store the information
        self.obs[self.step] = transition.obs.copy()
        self.actions[self.step] = transition.action.copy()
        self.rewards[self.step] = transition.reward.copy()
        self.dones[self.step] = transition.done

        self.actions_log_prob[self.step] = transition.action_log_prob.copy()
        self.values[self.step] = transition.value.copy()

        self.step += 1

    # clearing the storage for the next set of transitions.
    def clear(self):
        self.step = 0

    def compute_advantages(self, last_values):
        next_advantages = 0
        for step in reversed(range(self.max_timesteps)):
            # last_values: next_obs
            if step == self.max_timesteps - 1:
                next_values = last_values  # V(st+1)
            else:
                next_values = self.values[step + 1]  # V(st+1)
            next_is_not_terminate = 1.0 - self.dones[step]
            # check if is the "step" is healthy or unhealthy, if healthy  next_is_not_terminate = 1 - 0, if unhealthy
            # next_is_not_terminate = 1-1 = 0

            #GAE implementation
            # delta(t) = r(t) + gamma*V(st+1) - V(s) = A(st, at) = Q(s,a) - V(s) = A(st, at)
            delta = self.rewards[step] + next_is_not_terminate * self.gamma * next_values - self.values[step]
            # self.advantages[step] = delta
            # A(t) = delta(t) +  gamma(t)*lambda(t)*A(t+1)
            next_advantages = delta + self.gamma * self.lambda_ * next_advantages * next_is_not_terminate
            self.advantages[step] = next_advantages

    def compute_returns(self, last_values):
        self.compute_advantages(last_values)
        self.returns = self.advantages + self.values

    def mini_batch_generator(self, num_batches, num_epochs=8, device="cpu"):
        batch_size = self.max_timesteps // num_batches
        indices = np.random.permutation(num_batches * batch_size)

        obs = torch.from_numpy(self.obs).to(device).float()
        actions = torch.from_numpy(self.actions).to(device).float()
        values = torch.from_numpy(self.values).to(device).float()
        advantages = torch.from_numpy(self.advantages).to(device).float()
        old_actions_log_prob = torch.from_numpy(self.actions_log_prob).to(device).float()
        critic_observations = obs
        returns = torch.from_numpy(self.returns).to(device).float()


        for epoch in range(num_epochs):
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_idx = indices[start:end]

                obs_batch = obs[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                returns_batch = returns[batch_idx]
                yield (obs_batch, actions_batch, target_values_batch, advantages_batch, old_actions_log_prob_batch,
                       critic_observations_batch, returns_batch )
