import torch
import torch.nn as nn
from networks.networks import MLP
from torch.distributions import Normal


# MLP:Custom multi-layer perceptron (MLP)
# Following class defines two networks: actor N and critic N

# ACTOR NETWORK: provides a policy for selecting actions => METHOD: update_distribution()
# the standard deviation is learned during training.
# sample of actions for given policy=> METHOD: act()


# in case we already have trained model: we can use the act_inference method to get actions
# in case of training => we need to update parameters => use update_distribution

class ActorCritic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=512,  # Number of units in the hidden layers of the neural networks
                 n_layers=2,  # Number of hidden layers in the neural networks
                 init_std=1.0  # Initial standard deviation for the action distribution
                 ):
        super().__init__()
        # Initializes two neural networks: actor & network
        # The actor network that outputs the mean of the action distribution.
        self.actor = MLP(dim_in=state_dim,
                         dim_hidden=hidden_dim,
                         dim_out=action_dim,
                         n_layers=n_layers,
                         act=nn.ELU(),
                         output_act=nn.Tanh(),
                         using_norm=False)
        # The critic network that outputs the estimated value of the state.
        self.critic = MLP(dim_in=state_dim,
                          dim_hidden=hidden_dim,
                          dim_out=1,
                          n_layers=n_layers,
                          act=nn.ELU(),
                          output_act=None,
                          using_norm=False)

        # Action distribution
        self.std = nn.Parameter(init_std * torch.ones(action_dim))
        self.distribution = None

    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # actor update direct policy => update mean value with observations
    # Updates the action distribution based on the given observations using the actor network.
    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self.std)  # Gaussian Policy, pi_tetha(s,a)

    # returns actions for given policy
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    # Computes the log probability
    def get_actions_log_prob(self, actions):
        #  objection function? sum(gradient(logppi)) ?
        return self.distribution.log_prob(actions).sum(dim=-1)

    # making inferences without computing gradients.
    # for situations where you have a trained model,
    # and you want to use it to obtain actions without updating the model's parameters (weights and biases).
    def act_inference(self, observations):
        with torch.no_grad():
            actions_mean = self.actor(observations)

        return actions_mean

    # it does not keep track of the operations for backpropagation.

    # Evaluates the value of the state using the critic network.
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
