import os.path
from env.go_env import GOEnv
from rl_modules.storage import Storage
import torch
import torch.nn as nn
import torch.optim as optim
from rl_modules.actor_critic import ActorCritic


class RLAgent(nn.Module):
    def __init__(self,
                 env: GOEnv,
                 storage: Storage,
                 actor_critic: ActorCritic,
                 lr=1e-3,  # learning rate for an Adam optimizer
                 value_loss_coef=0.9,
                 num_batches=1,
                 num_epochs=1,
                 device='cpu',
                 action_scale=0.5
                 ):
        super().__init__()
        self.env = env
        self.storage = storage
        self.actor_critic = actor_critic
        self.num_batches = num_batches  # Number of batches used in each update (default is 1).
        self.num_epochs = num_epochs
        self.value_loss_coef = value_loss_coef  # Coefficient for the value loss term in the total loss (default is 1.0).
        self.device = device
        self.action_scale = action_scale  # Scaling factor for the action
        self.transition = Storage.Transition()  # to store information about each transition.
        # create the normalizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # OOP parameters
        self.epsilon_clip = 0.2
        self.entropy_coef = 0.001
        self.max_grad_norm = 1.0

        # Uses the actor_critic to calculate an action, value, and action log probability.

    def act(self, obs):

        # calculate action from actor_critic act => returns updated distribution (policy) by using observation
        action = self.actor_critic.act(obs).squeeze()
        self.transition.action = action.detach().cpu().numpy()
        self.transition.value = self.actor_critic.evaluate(obs).squeeze().detach().cpu().numpy()
        self.transition.action_log_prob = self.actor_critic.get_actions_log_prob(action).detach().cpu().numpy()
        return self.transition.action

    # Performs inference using the actor_critic without gradient computation.
    def inference(self, obs):
        return self.actor_critic.act_inference(obs).squeeze().detach().cpu().numpy()

    # Stores the transition in the storage.
    def store_data(self, obs, reward, done):
        self.transition.obs = obs
        self.transition.reward = reward
        self.transition.done = done

        # Record the transition
        self.storage.store_transition(self.transition)
        self.transition.clear()

    # takes the last observation Uses the actor_critic to compute the value of the last observation.
    def compute_returns(self, last_obs):
        last_values = self.actor_critic.evaluate(last_obs).detach().cpu().numpy()
        return self.storage.compute_returns(last_values)

    # update: using data from storage => to improve policy (updates the model parameters using gradient descent), calculate mean_value_loss, mean_actor_loss
    def update(self):
        mean_value_loss = 0
        mean_actor_loss = 0
        generator = self.storage.mini_batch_generator(self.num_batches, self.num_epochs,
                                                      device=self.device)  # get data from storage

        torch.autograd.set_detect_anomaly(True)

        for obs_batch, actions_batch, target_values_batch, advantages_batch, old_actions_log_prob_batch, critic_observations_batch, returns_batch  in generator:
            # Normalize the advantages not sure if we should do that
            # advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.actor_critic.act(obs_batch)  # update distribution => evaluate policy
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            entropy_batch = self.actor_critic.entropy

            # surrogate function
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = ratio * advantages_batch
            surrogate_clipped = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages_batch
            surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

            # Value function loss, for the definition check: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ part 9.
            value_batch = self.actor_critic.evaluate(critic_observations_batch) #?
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.epsilon_clip,self.epsilon_clip)

            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()

            # compute losses => using calculated advantages function

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            critic_loss = value_loss
            actor_loss = surrogate_loss

            print(f'critic_loss: {critic_loss}')
            print(f'actor_loss: {actor_loss}')

            # Gradient step
            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += critic_loss.item()
            mean_actor_loss += actor_loss.item()

        num_updates = self.num_epochs * self.num_batches
        mean_value_loss /= num_updates
        mean_actor_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_actor_loss

    # in each iterration calculate predicted action, from this prediction calculate step => rewards and next observations
    # Resets the environment and plays the game for a certain number of steps.
    def play(self, is_training=True, early_termination=True):  # ROLLOUT PHASE => collecting data
        obs, _ = self.env.reset()
        infos = []
        for _ in range(self.storage.max_timesteps):
            obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(dim=0)
            with torch.no_grad():
                if is_training:  # collect data
                    action = self.act(
                        obs_tensor)  # sample an action from policy => predicted action for the step calculation
                else:
                    action = self.inference(obs_tensor)
            obs_next, reward, terminate, info = self.env.step(action * self.action_scale)  # perform one step action
            infos.append(info)
            if is_training:  # store data
                self.store_data(obs, reward, terminate)  # collect and store data
            if terminate and early_termination:
                obs, _ = self.env.reset()
            else:
                obs = obs_next
        if is_training:  # prepare data for update
            self.compute_returns(torch.from_numpy(obs_next).to(self.device).float())

        return infos

    # learn: play (collect and store data) and update the policy (improve the policy with collected data)
    def learn(self, save_dir, num_learning_iterations=1000, num_steps_per_val=50):
        for it in range(num_learning_iterations):
            # play games to collect data => ROLLOUT PHASE
            infos = self.play(is_training=True)


            # improve policy with collected data => LEARNING PHASE => agent learns from collected data in Rollout phase
            mean_value_loss, mean_actor_loss = self.update()
            print(f"mean value loss: {mean_value_loss}")
            print(f"mean actor loss: {mean_actor_loss}")

            if it % num_steps_per_val == 0:
                infos = self.play(is_training=False)
                self.save_model(os.path.join(save_dir, f'{it}.pt'))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
