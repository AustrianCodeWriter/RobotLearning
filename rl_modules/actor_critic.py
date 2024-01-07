import torch
import torch.nn as nn
from networks.networks import MLP
from torch.distributions import Normal
import numpy as np


"""
   This interface defines a general model of an agent. It represents the Q(s,a) function and can be trained and used to predict stuff
   ActorCritic class defines the neural networks and methods for policy and value function updates
 """
class ActorCritic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=512,
                 n_layers=2,
                 init_std=1.0 #Changed from 1.0
                 ):
        super().__init__()
        self.actor = MLP(dim_in=state_dim,
                         dim_hidden=hidden_dim,
                         dim_out=action_dim, ##Passing action at output : The actor network that outputs the mean of the action distribution
                         n_layers=n_layers,
                         act=nn.ELU(),
                         output_act=nn.Tanh(),
                         using_norm=False)

        self.critic = MLP(dim_in=state_dim,
                          dim_hidden=hidden_dim,
                          dim_out=1, ## Passing value at output Q : The critic network that outputs the estimated value of the state.
                          n_layers=n_layers,
                          act=nn.ELU(),
                          output_act=None,
                          using_norm=False)


        # Action distribution: Initializes the standard deviation parameter for the action distribution.
        #Gaussian policy: model a distribution over continuous actions.
        self.std = nn.Parameter(init_std * torch.ones(action_dim))
        self.distribution = None

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        '''
        entropy is often used as a regularization term in the objective function to encourage exploration by adding diversity to the
        actions chosen by the policy. It helps prevent the policy from becoming overly deterministic and encourages exploration of
        different actions.
        '''
        return self.distribution.entropy().sum(dim=-1)

    def calculate_entropy(self, prob_actions):
        epsilon = 1e-12  # A small epsilon value to prevent log(0)
        prob_actions = torch.clamp(prob_actions, epsilon, 1.0)  # Clip probabilities to avoid log(0)
        return -torch.sum(prob_actions * torch.log(prob_actions), dim=-1)  # Calculate entropy along dim=-1

    # Adding Noise to Continuous Actions
    def add_noise(self, action, noise_scale):
        noise = np.random.normal(0, noise_scale)
        return action + noise

    def update_distribution(self, observations): #Observation 35 dim
        '''
         policy network might output parameters of a probability distribution, such as the mean and variance of a Gaussian (Normal) distribution,
         representing continuous actions.
        '''
        mean = self.actor(observations)

     #   mean = self.add_noise(mean, 0.2)  # Add noise for exploration
        self.distribution = Normal(mean, self.std) #Output cont. actions 12 dim

    def act(self, observations, **kwargs):
        '''
        PI policy is represented as a probability distribution over available actions for a given state
                observations: the current state
        A2C: interpreting how the policy generates actions probabilistically based on the observed state,
        guiding the agent's decision-making process.
        '''
        self.update_distribution(observations)
        return self.distribution.sample() #output is an action (12 dim)

    def get_actions_log_prob(self, actions):

        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations): #making predictions or selecting actions based on the learned policies and value estimates
        with torch.no_grad():
            actions_mean = self.actor(observations)

        return actions_mean

# Estimated value function is referred to as the critic, which evaluates actions taken by the actor based on the given policy
# Evaluates the value of the state using the critic network
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)  # Critic belongs to NN
        return value
# MODEL = fit and predict(value from critic - calculate q (apppy action and get state)) # Q = average(returns(s,a))

#1. actor provides a policy for selecting actions - action distribution
#The actor outputs a Gaussian distribution over actions with a mean provided by the neural network, and the standard deviation is learned during training
#2. critic estimates the value of states to guide the learning process