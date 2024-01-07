import numpy as np
import torch


"""Critic network -- approximate V
 The Storage class handles the storage and computation of returns and advantages
   A buffer for storing trajectory data and calculating returns for the policy
   and critic updates.
   """
class Storage:
    class Transition:
        def __init__(self):
            self.obs = None
            self.action = None
            self.reward = None
            self.done = None
            self.value = None
            self.action_log_prob = None

        def clear(self):
            self.__init__()

    def __init__(self,
                 obs_dim,
                 action_dim,
                 max_timesteps,
                 gamma=0.99,
                 lam=0.9):

        self.max_timesteps = max_timesteps
        self.gamma = gamma
        self.lam = lam

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
        self.ptr = 0
        self.traj_idx = [0]

    def store_transition(self, transition: Transition):
        # store the information
        self.obs[self.step] = transition.obs.copy()
        self.actions[self.step] = transition.action.copy()
        self.rewards[self.step] = transition.reward.copy()
        self.dones[self.step] = transition.done

        self.actions_log_prob[self.step] = transition.action_log_prob.copy()
        self.values[self.step] = transition.value.copy()

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_advantages(self, last_values):
       # self.returns = [0] * len(self.rewards)
        for step in reversed(range(self.max_timesteps)):
            if step == self.max_timesteps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminate = 1.0 - self.dones[step]
            delta = self.rewards[step] + next_is_not_terminate * self.gamma * next_values - self.values[step]
           # gae = delta + next_is_not_terminate * self.gamma * self.lam * gae
         #   self.advantages[step] = gae
          #  self.returns[step] =  self.advantages[step]
         #   print(delta)
            self.advantages[step] = delta
            self.returns[step] = self.advantages[step]
       # return delta
          #  self.returns[step]      = self.rewards[step] + next_is_not_terminate * self.gamma * next_values

#TODO compute_n_step_returns
    def compute_returns(self, last_values):
        """
 Computes returns and advantages for each timestep in reverse order using the Generalized Advantage Estimation (GAE) formula.
• GAE introduces a trade-off parameter to balance bias and variance in the advantage estimation
Advantages represent the estimated advantage of taking a specific action in a
specific state over a baseline (the value function estimate).

 If there is some advantage in choosing action  𝑎𝑡 over just following our policy
    - If choosing  𝑎𝑡 is better than following our policy, then  𝐴𝜋𝜃(𝑠𝑡,𝑎𝑡)>0. If it's worse than following our policy, then  𝐴𝜋𝜃(𝑠𝑡,𝑎𝑡)<0
    - For continuing environments, we can also get away with only fitting  𝑉𝜙′. Define the temporal-difference error or TD error  𝛿𝜋𝜃𝑡 to be:
             #𝛿𝜋𝜃𝑡=𝑟𝑡+𝛾𝑉𝜋𝜃(𝑠𝑡+1)−𝑉𝜋𝜃(𝑠𝑡)

Use in rl_agent module:
#1. use advantages to fit the value network  𝑉𝜙′ - Critic loss calculation
#2. use advantages to get an advantage estimate using GAE - Actor loss calculation
        """
      #  self.traj_idx += [self.ptr]
     #   rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

       # returns = []

      #  R = last_values  # Avoid copy?
       # for reward in reversed(rewards):
        #    R = self.gamma * R + reward
         #   returns.insert(0, R)  # TODO: self.returns.insert(self.path_idx, R) ?
            # also technically O(k^2), may be worth just reversing list
            # BUG? This is adding copies of R by reference (?)

          #  self.returns += returns

           # self.returns += [np.sum(rewards)]
      #  self.ep_lens += [len(rewards)]
        gae = 0
    #    advantages = []
       # self.returns = [0] * len(self.rewards)
        #Evaluate the value network:
        #Cumulative reward to be maximized for max_timesteps (steps in episode)
        for step in reversed(range(self.max_timesteps)):

            if step == self.max_timesteps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminate = 1.0 - self.dones[step] # By setting next_is_not_terminate to 0.0 for terminal states, subsequent reward estimations (such as bootstrapping) might be appropriately discounted or handled differently.
            #Actor maps obs to actions , Critic makes traininf efficient
            # Advantage function produced by critic network: A(s,t) = r(s,a) + V(s+1) - V(s) = Q(s,a) - V(s)
            # difference between the action value and the state value (actual return and the expected return)
           # best_next_action = torch.argmax(double_q, -1)
           # print(self.rewards[step])
            #The critic learns from reward signal: self.rewards[step] - update value policy
            #The critic supervises the actor to update: self.rewards[step] + next_is_not_terminate * self.gamma * next_values - delta
            #next value = q+1 = expected cumulative reward of taking a particular action
            #values = V(s) = expected cumulative reward from a particular state
            delta = self.rewards[step] + next_is_not_terminate * self.gamma * next_values - self.values[step] #Predicted/estimated by critic
            gae = delta + next_is_not_terminate * self.gamma * self.lam * gae #discounting future rewards and balancing bias-variance in GAE
            #return gae
            #advantages.append(gae)
            self.advantages[step] = gae
            self.returns[step] = self.advantages[step] + self.values[step]
            # self.advantages[step] = (self.advantages[step] - np.mean(self.advantages[step])) / (np.std(self.advantages[step]) + 1e-8)  # Normalize advantages (optional but can help reduce variance)
            #self.returns[step] = gae
            # print(f"TD_errors: {TD_errors}  ")

        # Compute GAE for advantages - example part 2
       # Actor_advantages = []
       #  gae = 0
       # for t in reversed(range(self.max_timesteps)):
        #    gae = TD_errors[t] + self.gamma * self.lam * gae #discounting future rewards and balancing bias-variance in GAE
           # Actor_advantages.append(gae)
           # print(f"GAE Advantages: {self.returns[t] } ")
         #   self.advantages[t] = gae
          #  self.returns[t] = self.advantages[t]
                              #+ self.values[t]


            #Add info: Update policy network: beta and theta
            #  DEFINE POLICY GRADIENT TODO A policy that controls how our agent acts:
            #Theta: If the advantage is positive, then we increase the log probability of the action  𝑎𝑡 associated with that advantage.
            #By using the advantage instead, we actually decrease the log-probability of actions that perform worse than expected.
            # Theta(t+1) = theta t + beta * qt * differentiate(theta t)








    def mini_batch_generator(self, num_batches, num_epochs, device="cpu"):
        batch_size = self.max_timesteps // num_batches
        indices = np.random.permutation(num_batches * batch_size)
        # get data of current state/action/values/advantages
        obs = torch.from_numpy(self.obs).to(device).float()
        actions = torch.from_numpy(self.actions).to(device).float()
        values = torch.from_numpy(self.values).to(device).float()
        advantages = torch.from_numpy(self.advantages).to(device).float()

        for epoch in range(num_epochs):
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_idx = indices[start:end]

                obs_batch = obs[batch_idx]
                actions_batch = actions[batch_idx]
             #   target_values_batch = values[batch_idx] #V(s)
                target_values_batch = values[batch_idx]
                advantages_batch = advantages[batch_idx] #A
               # target_values_batch = self.compute_returns(values[batch_idx])
               # advantages_batch = self.compute_advantages(advantages[batch_idx])
                yield (obs_batch, actions_batch, target_values_batch, advantages_batch)
