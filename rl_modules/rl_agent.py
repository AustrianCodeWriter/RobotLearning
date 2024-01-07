import os.path
from env.go_env import GOEnv
from rl_modules.storage import Storage
import torch
import torch.nn as nn
import torch.optim as optim
from rl_modules.actor_critic import ActorCritic
import numpy as np

###Critics can estimate future rewards but not completely because of randomness --> therefore instead of discounted return
##it calculate VALUE FUNCTION - used as baseline for performance (how well actor perform from given state)
## If value function worse then baseline - randomness is influenced bad -> diff of these values ADVANTAGE function (guidline how to improve actors actions)
## if possitive - encourage action associated with for each timestamp (x fraction - probability that actor take this action)
##
"""Defines the actor, who may act and learn""" #Change of these param inf. actor loss
class RLAgent(nn.Module):
    def __init__(self,
                 env: GOEnv,
                 storage: Storage,
                 actor_critic: ActorCritic,
                 lr=0.0001,
                 value_loss_coef=0.8, #lower -> critic loss higher- hyperparameter that influences the trade-off between improving the policy and accurately estimating the value function in A2C
                 num_batches=4,
                 num_epochs=2,
                 device='cpu',
                 action_scale=0.5
                 ):
        super().__init__()
        self.env = env
        self.storage = storage
        self.actor_critic = actor_critic
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.value_loss_coef = value_loss_coef
        self.device = device
        self.action_scale = action_scale
        self.transition = Storage.Transition()
        # create the normalizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        # Define optimizer for both actor and critic
       # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
       # self. critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def act(self, obs):
        """
        Given normalized state observation, return action given by network.
        Optional parameters to add or subtract variance to network weights when calculating action.
        """

        # Compute the actions and values ? MODEL IS ACTOR CRITIC - update position by applying action
        action = self.actor_critic.act(obs).squeeze()
        self.transition.action = action.detach().cpu().numpy()
        self.transition.value = self.actor_critic.evaluate(obs).squeeze().detach().cpu().numpy()
        self.transition.action_log_prob = self.actor_critic.get_actions_log_prob(action).detach().cpu().numpy()

        return self.transition.action

    def inference(self, obs):
        '''
        "inference" refers to the utilization of trained networks to make predictions or decisions in an environment
        without performing any updates or learning. Once the Actor and Critic networks are trained, inference involves
        using these networks to interact with the environment by making predictions or selecting actions based on the
        learned policies and value estimates.
        :param obs:
        :return:
        '''
        return self.actor_critic.act_inference(obs).squeeze().detach().cpu().numpy() ##actor part

    # Function store transition which stores all the transition information in the buffer for each time stamp
    # i.e state (obs), action, reward and we can use this transitions for training the network
    def store_data(self, obs, reward, done):
        self.transition.obs = obs
        self.transition.reward = reward
        self.transition.done = done

        # Record the transition
      #  print(f"Storage: {self.transition}")
      #  print(f"Reward: {reward}")
        self.storage.store_transition(self.transition)   ## Save pairs (observation, reward, done)
        self.transition.clear()
    #Critic learns
    def compute_returns(self, last_obs):
       # 3.Evaluate value network q+1 >> last values
        last_values = self.actor_critic.evaluate(last_obs).detach().cpu().numpy()
        return self.storage.compute_returns(last_values)  #Called from storage - evaluation using actor critic module param = critic_observations

    def compute_advantages(self, last_obs):
        last_values = self.actor_critic.evaluate(last_obs).detach().cpu().numpy()
        return self.storage.compute_advantages(last_values)


    def update(self):
        '''
        Actor network update: GRADIENT POLICY TO BE UPDATED improving the policy by increasing the probability of good actions and decreasing the probability of bad actions

    The ActorCritic is updated based on actor and critic losses calculated from the mini-batches
    Performs multiple updates using data from the storage.  updates the model parameters using gradient descent.
    Actor loss = policy gradient is calculated by multiplying the advantage function by the logarithmic probability of the action taken by the actor.
    Total loss improved by using various techniques, e.g. B. through entropy regularization, line subtraction or actor-criticism methods.

        '''
        mean_value_loss = 0
        mean_actor_loss = 0
        generator = self.storage.mini_batch_generator(self.num_batches, self.num_epochs, device=self.device)

        for obs_batch, actions_batch, target_values_batch, advantages_batch in generator:
            self.actor_critic.act(obs_batch) #predicted action
         #   print(advantages_batch)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
          #  print(actions_log_prob_batch)
         #   comp_returns = self.compute_returns(obs_batch).squeeze().detach().cpu().numpy()
           # advantages_batch = self.compute_advantages(obs_batch).squeeze().detach().cpu().numpy()
            # Weight log probabilities by advantages and Compute loss for policy update (negative weighted log probabilities):
            # 1. Update Actor network using advantage for policy improvement
          #  actor_loss = (-torch.log(actions_log_prob_batch) * advantages_batch).item()  # Negative because we want to maximize advantages
          #  real_adv = advantages_batch - target_values_batch
            actor_loss = -(actions_log_prob_batch * advantages_batch).mean() #negative ##TODO correct loss update
            # Advanatges = delta = positive the best
         #   actor_loss = -(actions_log_prob_batch * advantages_batch)

           # Compute TD error and update the Critic
           # td_target = self.storage.calculate_advantages(obs_batch)
           # Update value estimates using TD errors for the target values
           # target_values_batch =  target_values_batch + td_target

            # 2. Calculate loss for the Critic using TD error (MSE or other suitable loss)
            #Advantage = expected return - pred value func.
            # Critic loss = mse(predicted target, expected return)
            expected_return = advantages_batch + target_values_batch
            critic_loss = nn.MSELoss()(target_values_batch, expected_return)
           # critic_loss = nn.MSELoss()(self.actor_critic.evaluate(obs_batch), target_values_batch.detach())
           # critic_loss = expected_return.pow(2).mean() #positive


            # Calculate the policy entropy
            entropy_weight = 0.001
            # Calculate entropy based on the probabilities of actions
            entropy_val = self.actor_critic.calculate_entropy(actions_log_prob_batch).squeeze().detach().cpu().numpy()

           # Compute the total loss with entropy regularization
           # loss = actor_loss + critic_loss * self.value_loss_coef
            loss = actor_loss + self.value_loss_coef * critic_loss + entropy_val * entropy_weight
            print(f"actor loss: {actor_loss}  "
                  f"critic loss: {critic_loss}  "
                  f"total loss: {loss}  "
                  f"entropy: {entropy_val}  ")

            # Update network: Gradient step - calculate gradients with backwards and update with step
            #optimizer adjusts the weights in a direction that minimizes the value loss and maximizes the actor loss, thus improving the policy's performance
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step() # Perform gradient ascent by updating the actor's parameters in the direction that increases the expected rewards


            mean_value_loss += critic_loss.item()
            mean_actor_loss += actor_loss.item()


        #Update frequency is a hyperparameter that can affect the learning process and stability of the training
        num_updates = self.num_epochs * self.num_batches
        print(f"mean_value loss: {mean_value_loss}  "
              f"mean_actor loss: {mean_actor_loss}  "
              )
        #Update actor and critic network?
        mean_value_loss /= num_updates #how well the value function approximates the expected returns
        mean_actor_loss /= num_updates #aims to maximize the expected rewards by adjusting the policy towards actions that lead to higher advantages
        self.storage.clear()

        return mean_value_loss, mean_actor_loss

    #Computes returns at the end of the episode - CRITIC Part like MC?
    def play(self, is_training=True, early_termination=True):   ##Training - Learning purpose ## TODO make actions (play) - done in step in env
        obs, _ = self.env.reset()
        infos = []
        # rewards = []
        #Part for calculating next state, reward, info for each timestamp
        for _ in range(self.storage.max_timesteps):
            #1. Observe the state and randomly sample action
            obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(dim=0)
            with torch.no_grad():
                #The agent can use the updated policy to act in the environment (act method) or perform inference (inference method).
                if is_training:
                    #3. Evaluate q - This selection of action indirectly involves the evaluation of Q-values Action-Value function
                    action = self.act(obs_tensor)
                else:
                    action = self.inference(obs_tensor)  #evaluate without update - making predictions or selecting actions based on the learned policies and value estimates.

          # Choose action by sampling random action from environment's action space.
          # Every environment hassome action space which contains the all possible valid actions and observations,
          # For each step, we will record the observation, reward, done, info
          # Calculate action of act agent and use for rewards and compute returns
         # Evaluation part of critic to get obs_next ## TODO each action triggers rewards- which shall be corrected
            #LEARNING PART FOR CRITIC - learns from rewards
           # 2. Perform action then environment gives new state obs_next and the reward .
            obs_next, reward, terminate, info = self.env.step(action*self.action_scale)  #delta_q Step function to env: Apply actions, simulate - Actor NN
           # print(f"Total reward: {list(info.values())[0]}  ")
            infos.append(info)
            #Depending on action and state - for training stored and for testing used existing
            if is_training:
                #Stored data for each timestamp V(s)
                self.store_data(obs, reward, terminate)
            if terminate and early_termination:
                obs, _ = self.env.reset()
            else: #Testing go seq.
                obs = obs_next
        #Part for calculating average function used for update
        if is_training: #compute returns to the RL algorithm's reward calculation or the estimation of expected future rewards
            self.compute_returns(torch.from_numpy(obs_next).to(self.device).float())
            self.compute_advantages(torch.from_numpy(obs_next).to(self.device).float())
        #Infos are calculated for each update for each timestamp max_timesteps eg 50
        return infos




    def learn(self, save_dir, num_learning_iterations=1000, num_steps_per_val=100):
        """train internal model with new episode"""
        for it in range(num_learning_iterations):
            # play games to collect data
            infos = self.play(is_training=True)
            # Goal: encourage the policy to favor actions that lead to higher expected rewards,
            # as indicated by the advantages calculated from the collected experiences in previous step
            mean_value_loss, mean_actor_loss = self.update()  # UPDATE actor GRADIENT POLICY
            if it % num_steps_per_val == 0: #inference, primarily focused on using the learned policy for decision-making, is not typically used for actual training updates.
                infos = self.play(is_training=False)
                self.save_model(os.path.join(save_dir, f'{it}.pt'))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))



