import torch
from torch.distributions import Categorical


def collect_rollout(env, actor, critic, device):
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    obs, info = env.reset()

    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        obs_tensor = obs_tensor.unsqueeze(0)
        logits = actor(obs_tensor)
        value = critic(obs_tensor)

        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        env_action = action.item()
        next_obs, reward, terminated, truncated, info = env.step(env_action)

        done = terminated or truncated

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.detach())
        values.append(value.detach())

        obs = next_obs
    return observations, actions, rewards, dones, log_probs, values