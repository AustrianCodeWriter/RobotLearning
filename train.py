from datetime                   import datetime
from env.go_env                 import GOEnv
from rl_modules.actor_critic    import ActorCritic
from rl_modules.rl_agent        import RLAgent
from rl_modules.storage         import Storage
import os


def train():
    device ='cuda:0'
    log_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

    # create Golang environment
    go_env = GOEnv(render_mode="human")

    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim).to(device)

    # create storage to save data
    storage     = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, max_timesteps=1000)
    rl_agent    = RLAgent(env=go_env, actor_critic=actor_critic, storage=storage, device=device)

    save_dir    = f'checkpoints/{log_name}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rl_agent.learn(save_dir, num_learning_iterations=1000, num_steps_per_val=100)


if __name__ == '__main__':
    train()