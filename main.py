import gymnasium as gym
import torch
from gae import calculate_gae, calculate_returns
from rollout import collect_rollout
from agent import Actor, Critic
from ppo import ppo


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


env = gym.make("CartPole-v1")

actor = Actor().to(device)
critic = Critic().to(device)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

for iteration in range(1000):
    rollout = collect_rollout(env, actor, critic, device)


    observations, actions, rewards, dones, log_probs, values = rollout
    episode_reward = sum(rewards)
    if iteration % 50 == 0:
        print(f"Iteration {iteration}: reward = {episode_reward}")

    gae = calculate_gae(rewards, values, dones, 0.95, 0.99)
    returns = calculate_returns(gae, values)
    new_log_probs, ratio, ppo_objective, actor_loss, critic_loss = ppo(
        actor,
        critic,
        observations,
        actions,
        log_probs,
        gae,
        returns,
        0.2
        )

    

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

print("Observations:", len(observations))
print("Actions:", len(actions))
print("Rewards:", len(rewards))
print("Dones:", len(dones))
print("Log probs:", len(log_probs))
print("Values:", len(values))

torch.save(actor.state_dict(), "actor.pth")

env.close()