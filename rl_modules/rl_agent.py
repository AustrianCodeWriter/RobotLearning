import  os.path
from env.go_env import GOEnv
from rl_modules.actor_critic    import ActorCritic
from rl_modules.storage         import Storage
import torch
import torch.nn as nn
import torch.optim as optim



class RLAgent(nn.Module):
    def __init__(self,
                 env:           GOEnv,
                 storage:       Storage,
                 actor_critic:  ActorCritic,
                 lr             = 1e-3,
                 value_loss_coef= 1.0,
                 num_batches    = 1,
                 num_epochs     = 1,
                 device         ='cpu',
                 action_scale   = 0.3
                 ):
        super().__init__()
        self.env                = env
        self.storage            = storage
        self.actor_critic       = actor_critic
        self.num_batches        = num_batches
        self.num_epochs         = num_epochs
        self.value_loss_coef    = value_loss_coef
        self.device             = device
        self.action_scale       = action_scale
        self.transition         = Storage.Transition()

        # create the normalizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def act(self, obs):
        # Compute the actions and values
        action                          = self.actor_critic.act(obs).squeeze()
        self.transition.action          = action.detach().cpu().numpy()
        self.transition.value           = self.actor_critic.evaluate(obs).squeeze().detach().cpu().numpy()
        self.transition.action_log_prob = self.actor_critic.get_actions_log_prob(action).detach().cpu().numpy()
        return self.transition.action

    def inference(self, obs):
        return self.actor_critic.act_inference(obs).squeeze().detach().cpu().numpy()

    def store_data(self, obs, reward, done):
        self.transition.obs     = obs
        self.transition.reward  = reward
        self.transition.done    = done

        # Record the transition
        self.storage.store_transition(self.transition)
        self.transition.clear()

    def compute_returns(self, last_obs):
        last_values = self.actor_critic.evaluate(last_obs).detach().cpu().numpy()
        return self.storage.compute_returns(last_values)

    def update(self):
        mean_value_loss = 0
        mean_actor_loss = 0
        generator = self.storage.mini_batch_generator(self.num_batches, self.num_epochs, device=self.device)

        for obs_batch, actions_batch, target_values_batch, advantages_batch in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            actor_loss = (-advantages_batch * actions_log_prob_batch).mean()
            critic_loss = advantages_batch.pow(2).mean()
            loss = actor_loss + self.value_loss_coef * critic_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_value_loss += critic_loss.item()
            mean_actor_loss += actor_loss.item()

        num_updates = self.num_epochs * self.num_batches
        mean_value_loss /= num_updates
        mean_actor_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_actor_loss

    def play(self, is_training=True, early_termination=True):
        obs, _ = self.env.reset()
        infos = []
        for _ in range(self.storage.max_timesteps):
            obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(dim=0)
            with torch.no_grad():
                if is_training:
                    action = self.act(obs_tensor)
                else:
                    action = self.inference(obs_tensor)
            obs_next, reward, terminate, info = self.env.step(action*self.action_scale)
            infos.append(info)
            if is_training:
                self.store_data(obs, reward, terminate)
            if terminate and early_termination:
                obs, _ = self.env.reset()
            else:
                obs = obs_next
        if is_training:
            self.compute_returns(torch.from_numpy(obs_next).to(self.device).float())

        return infos

    def learn(self, save_dir, num_learning_iterations=1000, num_steps_per_val=50):
        for it in range(num_learning_iterations):
            # play games to collect data
            self.play(is_training=True)

            # improve policy with collected data
            self.update()

            if it % num_steps_per_val == 0:
                self.play(is_training=False)
                self.save_model(os.path.join(save_dir, f'{it}.pt'))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))