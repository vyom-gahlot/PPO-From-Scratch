import torch
from torch.distributions import Categorical


def ppo(actor, critic, observations, actions, old_log_probs, advantages, returns, epsilon):

    new_log_probs = []
    new_values = []
    ratio = []

    # Get new policy probabilities and new value predictions
    for i in range(len(observations)):

        obs_tensor = torch.tensor(observations[i], dtype=torch.float32)
        obs_tensor = obs_tensor.unsqueeze(0)

        # Actor
        logits = actor(obs_tensor)

        # Critic
        new_value = critic(obs_tensor)
        new_values.append(new_value)

        # Policy distribution
        distribution = Categorical(logits=logits)

        # Probability of the action that was actually taken
        log_prob = distribution.log_prob(actions[i])
        new_log_probs.append(log_prob)

    # Probability ratio
    for i in range(len(old_log_probs)):

        ratio.append(
            torch.exp(new_log_probs[i] - old_log_probs[i])
        )

    # PPO clipped objective
    L1 = []
    L2 = []
    ppo_objective = []

    for i in range(len(ratio)):

        L1.append(
            ratio[i] * advantages[i]
        )

        L2.append(
            torch.clamp(
                ratio[i],
                1 - epsilon,
                1 + epsilon
            ) * advantages[i]
        )

    for i in range(len(L1)):

        ppo_objective.append(
            torch.minimum(L1[i], L2[i])
        )

    ppo_objective = torch.stack(ppo_objective)

    # Actor loss
    L_clip = 0

    for i in range(len(ppo_objective)):

        L_clip = L_clip + ppo_objective[i]

    L_clip = L_clip / len(ppo_objective)

    actor_loss = -L_clip

    # Critic loss
    critic_error = 0

    for i in range(len(new_values)):

        squared_error = (
            new_values[i] - returns[i]
        ) ** 2

        critic_error += squared_error

    critic_loss = critic_error / len(new_values)

    return (
        new_log_probs,
        ratio,
        ppo_objective,
        actor_loss,
        critic_loss
    )
