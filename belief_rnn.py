import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os







class BeliefRNN(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # RNN to process the history of observations and actions
        self.rnn = nn.GRU(input_dim, latent_dim, batch_first=True)

    def forward(self, history):
        z, _ = self.rnn(history)  # [B, T, H]

        return z













class ValueReadout(nn.Module):
    def __init__(self, latent_dim, hidden=128):
        super().__init__()
        # non-linear transformation mapping b -> w
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, b):
        w = self.net(b)                 # shape (batch, d)
        out = torch.sum(w * b, dim=-1)  # inner product <w, b>
        
        return out
    
    def reveal_w(self, b):
        return self.net(b)

# TD loss function
def loss_value_td(values, rewards, mask_traj, lengths, gamma=1.0):
    
    # calculate the TD target
    values_next = torch.zeros_like(values)
    values_next[:, :-1] = values[:, 1:]

    # Zero out bootstrap at terminal state
    for b, l in enumerate(lengths): 
        values_next[b, l-1] = 0.0

    # TD target
    td_target = rewards + (gamma * values_next.detach())

    # Squared TD error
    td_error = (values - td_target)**2

    # Mask invalid positions and average loss over non-masked values
    td_error = td_error * mask_traj
    loss = td_error.sum() / mask_traj.sum()

    return loss






class QReadout(nn.Module):
    """
    Map latent z_t to Q-values Q(z_t, a) for all actions.
    
    Input:
        z : [B, T, latent_dim] or [B, latent_dim]
    Output:
        Q : [B, T, A] or [B, A]  (broadcast over extra dims)
    """
    def __init__(self, latent_dim, num_actions, hidden=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )

    def forward(self, z):
        # Works for [B, T, H] or [B, H] transparently
        return self.net(z)  # last dim becomes num_actions

    def q_for_actions(self, z, actions_one_hot):
        """
        Convenience helper:
            z              : [B, T, latent_dim]
            actions_one_hot: [B, T, A] (one-hot)
        Returns:
            Q(z_t, a_t)    : [B, T]
        """
        q_all = self.forward(z)              # [B, T, A]
        q_sa  = torch.sum(q_all * actions_one_hot, dim=-1)
        return q_sa


def loss_q_td(
        q_values:   torch.Tensor,  # [B, T, A] = Q(z_t, a)
        rewards:    torch.Tensor,  # [B, T]    = r_t  (reward at state index t)
        actions:    torch.Tensor,  # [B, T, A] = one-hot prev_action
        mask_traj:  torch.Tensor,  # [B, T]    = 1 for valid, 0 for padded
        lengths:    list,
        gamma:      float = 1.0
    ) -> torch.Tensor:
    """
    TD(0) loss for Q-values with your indexing convention.

    We train Q(z_t, a_t) using:
        target_t = r_{t+1} + gamma * max_{a'} Q(z_{t+1}, a')
    where:
        - z_t      is at index t
        - a_t      is stored at actions[:, t+1]
        - r_{t+1}  is rewards[:, t+1]
        - z_{t+1}  is at index t+1

    The loss is averaged over all valid transitions (excluding padding).
    """

    # ----- Slice to transitions -----
    # Use time dimension T_full, but transitions exist only for t = 0..T-2
    # We'll index them via "j" = t+1 in the original arrays.

    # Q(z_t, ·) for t=0..T-2
    q_t_all   = q_values[:, :-1, :]       # [B, T-1, A]

    # a_t is stored at index j = t+1 → slice actions[:, 1:, :]
    actions_tp1 = actions[:, 1:, :]       # [B, T-1, A]

    # r_{t+1} is at index j = t+1
    r_tp1    = rewards[:, 1:]             # [B, T-1]

    # Mask for those time indices
    mask_tp1 = mask_traj[:, 1:]           # [B, T-1]

    # Q(z_{t+1}, ·) at index j = t+1
    q_next_all = q_values[:, 1:, :]       # [B, T-1, A]

    # ----- Gather Q(z_t, a_t) -----
    q_sa = torch.sum(q_t_all * actions_tp1, dim=-1)   # [B, T-1]

    # ----- Bootstrap: V_next = max_a' Q(z_{t+1}, a') -----
    V_next, _ = q_next_all.max(dim=-1)                # [B, T-1]

    # Zero out bootstrap at terminal next-state
    # For an episode of length L, the last transition is t = L-2
    # which corresponds to index i = L-2 in these [T-1] tensors.
    for b, L in enumerate(lengths):
        if L >= 2:
            V_next[b, L-2] = 0.0

    # ----- TD target -----
    td_target = r_tp1 + gamma * V_next.detach()       # [B, T-1]

    # ----- Squared TD error -----
    td_error = (q_sa - td_target) ** 2                # [B, T-1]

    # Mask out padded positions
    td_error = td_error * mask_tp1

    # Avoid divide-by-zero if something weird happens
    denom = mask_tp1.sum()
    if denom.item() == 0:
        return td_error.mean() * 0.0  # safe 0-loss

    loss = td_error.sum() / denom
    return loss



class RewardReadout(nn.Module):
    """
    Map the latent z_t to a prediction for the reward r_t
    """
    
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 1)

    def forward(self, z):
        x = self.fc1(z)
        
        return x.squeeze(-1)  # [B, T]


def loss_reward(
        est_rewards:    torch.tensor, 
        rewards:        torch.tensor, 
        mask:           torch.tensor,
        pred_steps:     int
        ):
    """
    Calculate the mean squared reward estimation loss
    """
    # Compute the reward loss
    rewards = rewards[:,pred_steps:]
    mask    = mask[:,pred_steps:]
    
    reward_loss = F.mse_loss(est_rewards, rewards, reduction='none') * mask  # sum over valid time steps
    return reward_loss.sum() / mask.sum()  # average over non-masked values



class NextLatentPredictor(nn.Module):
    """
    Predict the next latent state given the current z_t and a_t
    """
    def __init__(self, latent_dim, hidden=64):
        super().__init__()
        # non-linear transformation mapping b -> w
        self.net = nn.Sequential(
            nn.Linear(latent_dim+4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, z, a, pred_steps):
       
        pred_input = z[:, :-1, :]       # h_t
        next_actions = a[:, pred_steps:, :]      # a_{t + pred_steps}

        predictor_input = torch.cat([pred_input, next_actions], dim=-1)  # [B, T - pred_steps, Z+4]
        
        pred = self.net(predictor_input)

        return pred




class ObsReadout(nn.Module):
    """
    What is the distribution over observations given the latent (returns logits)
    """
    def __init__(self, latent_dim, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, obs_dim)

    def forward(self, z):
        x = self.fc1(z)
        
        return x

def loss_obs(
        est_o_logits    : torch.tensor, 
        o               : torch.tensor, 
        mask            : torch.tensor, 
        pred_steps      : int
        ):
    """
    Categorical cross entropy loss for observation prediction
    """

    pred_obs_target = o[:, pred_steps:, :]        # o_{t+1}
    aux_mask        = mask[:, pred_steps:]        # mask for t+1

    # Compute cross-entropy loss
    logits          = est_o_logits.transpose(1, 2)
    target_labels   = pred_obs_target.argmax(dim=-1)  # [B, T - pred_steps]
    aux_loss = F.cross_entropy(logits, target_labels, reduction='none')  # [B, T - pred_steps]

    aux_loss = aux_loss * aux_mask  # Apply mask
    return aux_loss.sum() / aux_mask.sum()
    


class ActorReadout(nn.Module):
    """
    Stochastic policy network:
        z_t → hidden → logits → Categorical(pi)
    """
    def __init__(self, latent_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, z):
        """
        Returns action probabilities (softmax).
        z : [B, T, latent_dim]
        """
        h = F.relu(self.fc1(z))
        logits = self.fc2(h)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def sample_action(self, z_t):
        """
        z_t: [latent_dim] or [1, latent_dim]
        Returns: action index, log_prob
        """
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)

        probs = self.forward(z_t).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)