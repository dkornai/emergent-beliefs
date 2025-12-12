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
        est_rewards:   torch.tensor, 
        rewards:        torch.tensor, 
        mask:           torch.tensor
        ):
    """
    Calculate the mean squared reward estimation loss
    """
    # Compute the reward loss
    reward_loss = F.mse_loss(est_rewards, rewards, reduction='none') * mask  # sum over valid time steps
    
    return reward_loss.sum() / mask.sum()  # average over non-masked values





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


def loss_value_mc(values, returns, mask_monte_carlo):
    # Compute the mc loss only at start and terminal states
    mc_values = values * mask_monte_carlo
    mc_returns = returns * mask_monte_carlo
    
    # Set first state return to the average of all first states
    start_state_return = mc_returns[:, 0].mean()
    mc_returns[:, 0] = start_state_return  

    mc_error = (mc_values - mc_returns) ** 2

    mc_loss = mc_error.sum() / mask_monte_carlo.sum()  # average loss over non-masked values

    return mc_loss





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

    def forward(self, z, a):
       
        pred_input = z[:, :-1, :]       # h_t
        next_actions = a[:, 1:, :]      # a_{t+1}

        predictor_input = torch.cat([pred_input, next_actions], dim=-1)  # [B, T-1, Z+4]
        
        pred = self.net(predictor_input)

        return pred




class ObsReadout(nn.Module):
    """
    What is the distribution over observations given the latent
    """
    def __init__(self, latent_dim, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, obs_dim)

    def forward(self, z):
        x = self.fc1(z)
        
        return x




def loss_obs(est_o_logits, o, mask):
    # x includes actions in last 4 dims

    pred_obs_target = o[:, 1:, :]        # o_{t+1}
    aux_mask = mask[:, 1:]                 # mask for t+1

    # Compute cross-entropy loss
    target_labels = pred_obs_target.argmax(dim=-1)  # [B, T-1]
    aux_loss = F.cross_entropy(est_o_logits.transpose(1, 2), target_labels, reduction='none')  # [B, T-1]

    aux_loss = aux_loss * aux_mask  # Apply mask
    return aux_loss.sum() / aux_mask.sum()
    


