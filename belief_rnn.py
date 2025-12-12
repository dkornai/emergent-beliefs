import torch
import torch.nn as nn
import torch.nn.functional as F
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
    


