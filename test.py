from env.go_env                 import GOEnv
from rl_modules.actor_critic    import ActorCritic
from rl_modules.rl_agent        import RLAgent
from rl_modules.storage         import Storage


def test():
    device ='cuda:0'

    # create Golang environment
    go_env = GOEnv(render_mode="human")

    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim).to(device)

    # create storage to save data
    storage     = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, max_timesteps=1000)
    rl_agent    = RLAgent(env=go_env, actor_critic=actor_critic, storage=storage, device=device)

    # rl_agent.load_model('checkpoints/Trained1/100.pt')
    rl_agent.load_model('checkpoints/best.pt')
    rl_agent.play(is_training=False)


if __name__ == '__main__':
    test()