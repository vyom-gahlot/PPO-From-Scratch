def calculate_gae(rewards, values, dones, lam, gamma):

    advantages = []
    gae = 0

    last = len(rewards) - 1

    for t in range(last, -1, -1):

        if t == last:
            next_value = 0
        else:
            next_value = values[t + 1]

        td_error = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        gae = td_error + gamma * lam * (1 - dones[t]) * gae

        advantages.append(gae)

    advantages.reverse()

    return advantages

def calculate_returns(advantages, values):
    steps = len(advantages)
    returns = []

    for i in range(steps):
        returns.append(advantages[i] + values[i])
    return returns