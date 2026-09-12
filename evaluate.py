import gymnasium as gym
import torch
from torch.distributions import Categorical

from agent import Actor



actor = Actor()


actor.load_state_dict(
torch.load("actor.pth", weights_only=True)
)


actor.eval()


env = gym.make(
"CartPole-v1",
render_mode="human"
)

obs, info = env.reset()

done = False
total_reward = 0

while not done:

    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits = actor(obs_tensor)

    distribution = Categorical(logits=logits)

    action = distribution.sample().item()

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    total_reward += reward


print(f"Episode reward: {total_reward}")

env.close()
