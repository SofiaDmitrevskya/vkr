import torch.nn.functional as F

def domination_penalty_loss(x_hat, dominated_pairs):
    penalty = 0.0
    eps = 1e-6
    for (i, j) in dominated_pairs:
        violation = F.relu(x_hat[i] - x_hat[j] + eps).sum()
        penalty += violation
    return penalty / len(dominated_pairs)
