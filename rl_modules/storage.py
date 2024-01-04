import numpy as np
import torch


class Storage:
    class Transition:
        def __init__(self):
            self.obs            = None
            self.action         = None
            self.reward         = None
            self.done           = None
            self.value          = None
            self.action_log_prob= None

        def clear(self):
            self.__init__()

    def __init__(self,
                 obs_dim,
                 action_dim,
                 max_timesteps,
                 gamma=0.98):
        self.max_timesteps = max_timesteps
        self.gamma = gamma

        # create the buffer
        self.obs        = np.zeros([self.max_timesteps, obs_dim])
        self.actions    = np.zeros([self.max_timesteps, action_dim])
        self.rewards    = np.zeros([self.max_timesteps])
        self.dones      = np.zeros([self.max_timesteps])

        # For RL methods
        self.actions_log_prob = np.zeros([self.max_timesteps])
        self.values     = np.zeros([self.max_timesteps])
        self.returns    = np.zeros([self.max_timesteps])
        self.advantages = np.zeros([self.max_timesteps])

        self.step = 0

    def store_transition(self, transition: Transition):
        # store the information
        self.obs[self.step]     = transition.obs.copy()
        self.actions[self.step] = transition.action.copy()
        self.rewards[self.step] = transition.reward.copy()
        self.dones[self.step]   = transition.done

        self.actions_log_prob[self.step] = transition.action_log_prob.copy()
        self.values[self.step]  = transition.value.copy()

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values):
        for step in reversed(range(self.max_timesteps)):
            if step == self.max_timesteps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminate = 1.0 - self.dones[step]

            # Modification here!!
            delta = self.rewards[step] + next_is_not_terminate * self.gamma * next_values - self.values[step]
            self.advantages[step]   = delta
            self.returns[step]      = self.advantages[step] + self.values[step]

    def mini_batch_generator(self, num_batches, num_epochs=8, device="cpu"):
        batch_size = self.max_timesteps // num_batches
        indices = np.random.permutation(num_batches * batch_size)

        obs         = torch.from_numpy(self.obs).to(device).float()
        actions     = torch.from_numpy(self.actions).to(device).float()
        values      = torch.from_numpy(self.values).to(device).float()
        advantages  = torch.from_numpy(self.advantages).to(device).float()

        for epoch in range(num_epochs):
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_idx = indices[start:end]

                obs_batch           = obs[batch_idx]
                actions_batch       = actions[batch_idx]
                target_values_batch = values[batch_idx]
                advantages_batch    = advantages[batch_idx]
                yield (obs_batch, actions_batch, target_values_batch, advantages_batch)