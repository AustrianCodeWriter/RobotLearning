from rl_modules.rl_agent import RLAgent
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
import torch


def test():
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'
    # create environment
    go_env = GOEnv(render_mode="human")
    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim).to(device)
    # create storage to save data
    storage = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, max_timesteps=1000)
    rl_agent = RLAgent(env=go_env, actor_critic=actor_critic, storage=storage, device=device)

    #rl_agent.load_model('checkpoints/2023-11-29/16-35-42/best.pt')
    rl_agent.load_model('checkpoints/2024-01-06/22-43-32/900.pt')
    rl_agent.play(is_training=False)


if __name__ == '__main__':
    test()
