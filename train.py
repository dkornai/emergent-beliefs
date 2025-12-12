import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from episodes import EpisodeCollection
from belief_rnn import loss_value_td, loss_obs, loss_reward

# Training loop
def train(
        models            : list[nn.Module],
        train_episodes    : EpisodeCollection, 
        num_epochs        = 10, 
        gamma             = 1.0, 
        lr                = 1e-3, 
        batch_size        = 32, 
        lambda_td         = 1.0,
        lambda_opl        = 1.0, 
        lambda_rel        = 1.0, 
        lambda_rpl        = 1.0, 
        ):
    
    # Unpack models
    belief_model, value_model, reward_model, pred_model, obs_model = models

    # Set up input and target tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
    
    assert type(train_episodes) == EpisodeCollection, f"episodes must be an EpisodeCollection object, not {type(train_episodes)}"
    histories       = train_episodes.batch_histories.to(device) 
    observations    = train_episodes.batch_observations.to(device)
    actions         = train_episodes.batch_actions.to(device)
    rewards         = train_episodes.batch_rewards.to(device) 
    mask_traj       = train_episodes.batch_mask_traj.to(device)  
    ep_lengths = train_episodes.ep_lengths

    # Validate dimensions
    assert belief_model.input_dim == histories.shape[-1], f"Belief model input dimension {belief_model.input_dim} does not match history dimension {histories.shape[-1]}"

    # Send all models to the GPU
    belief_model.to(device)
    value_model.to(device)
    reward_model.to(device)
    pred_model.to(device)
    obs_model.to(device)
    
    # Set up optimisers
    optimizer = torch.optim.Adam(
        list(belief_model.parameters()) + \
        list(value_model.parameters())  + \
        list(reward_model.parameters()) + \
        list(pred_model.parameters())   + \
        list(obs_model.parameters()),
        lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    # Set up loss histories
    lh_value_td = []
    lh_rew_est  = []
    lh_obs_pred_1 = []
    lh_rew_pred_1 = []
    lh_obs_pred_2 = []
    lh_rew_pred_2 = []
    loss_histories = lh_value_td, lh_rew_est, lh_rew_pred_1, lh_obs_pred_1, lh_rew_pred_2, lh_obs_pred_2

    # Training loop 
    for epoch in range(num_epochs):
        
        # Choose "batch_size" random episodes from the dataset
        chosen_i = np.random.choice(len(train_episodes), size=min(batch_size, len(train_episodes)), replace=False)
        batch_lengths = [ep_lengths[i] for i in chosen_i]


        optimizer.zero_grad()
        # Calculate latents 
        z = belief_model(histories[chosen_i])  # shape: [B, T]
        # Estimate values
        est_values = value_model(z)
        # Estimate rewards
        est_rewards = reward_model(z)
        
        # Predict next latents
        pred_z_1 = pred_model(z, actions[chosen_i], pred_steps = 1)
        # Predict next rewards
        pred_rewards_1 = reward_model(pred_z_1)
        # Predict next observation
        pred_obs_1 = obs_model(pred_z_1)
        
        # Predict next latents
        pred_z_2 = pred_model(pred_z_1, actions[chosen_i], pred_steps = 2)
        # Predict next rewards
        pred_rewards_2 = reward_model(pred_z_2)
        # Predict next observation
        pred_obs_2 = obs_model(pred_z_2)

        
        # Calculate TD error loss
        if lambda_td > 0.0:
            l_value_td = loss_value_td(est_values, rewards[chosen_i], mask_traj[chosen_i], batch_lengths, gamma) * lambda_td
        else:
            l_value_td = torch.tensor(0.0)

        # Calculate reward estimation loss
        if lambda_rel > 0.0:
            l_rew_est = loss_reward(est_rewards, rewards[chosen_i], mask_traj[chosen_i], pred_steps=0) * lambda_rel
        else:
            l_rew_est = torch.tensor(0.0)

        # Calculate reward prediction loss
        if lambda_rpl > 0.0:
            l_rew_pred_1 = loss_reward(pred_rewards_1, rewards[chosen_i], mask_traj[chosen_i], pred_steps=1) * lambda_rpl
        else:
            l_rew_pred_1 = torch.tensor(0.0)

        # Calculate observation prediction loss
        if lambda_opl > 0.0:
            l_obs_pred_1 = loss_obs(pred_obs_1, observations[chosen_i], mask_traj[chosen_i], pred_steps=1) * lambda_opl
        else:
            l_obs_pred_1 = torch.tensor(0.0)

        # Calculate next reward prediction loss
        if lambda_rpl > 0.0:
            l_rew_pred_2 = loss_reward(pred_rewards_2, rewards[chosen_i], mask_traj[chosen_i], pred_steps=2) * lambda_rpl
        else:
            l_rew_pred_2 = torch.tensor(0.0)

        # Calculate next observation prediction loss
        if lambda_opl > 0.0:
            l_obs_pred_2 = loss_obs(pred_obs_2, observations[chosen_i], mask_traj[chosen_i], pred_steps=2) * lambda_opl
        else:
            l_obs_pred_2 = torch.tensor(0.0)

        loss = l_value_td + l_rew_est + l_rew_pred_1 + l_obs_pred_1 + l_rew_pred_2 + l_obs_pred_2
        
        # Backpropagation
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to avoid exploding gradients
        optimizer.step()
        scheduler.step()

        # Keep track of losses
        losses = l_value_td, l_rew_est, l_rew_pred_1, l_obs_pred_1, l_rew_pred_2, l_obs_pred_2
        to_print = loss_tracker(losses, loss_histories)

        print(f"Epoch {epoch+1} {to_print}", end= '\r')
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1} {to_print}", end= '\n')
        


    models = belief_model, value_model, reward_model, pred_model, obs_model

    return loss_histories, models

def loss_tracker(losses, loss_histories):
    """
    Updates loss histories and generates loss string
    
    :param losses: Description
    :param loss_histories: Description
    """
    l_value_td, l_rew_est, l_rew_pred_1, l_obs_pred_1, l_rew_pred_2, l_obs_pred_2 = losses

    # Round
    td_L = np.round(l_value_td.item(), 2) 
    rew_L = np.round(l_rew_est.item(), 2)
    obs_L = np.round(l_obs_pred_1.item(), 2) 
    rew_pL = np.round(l_rew_pred_1.item(), 2)
    obs_L2 = np.round(l_obs_pred_2.item(), 2) 
    rew_pL2 = np.round(l_rew_pred_2.item(), 2)

    # Write to histories
    lh_value_td, lh_rew_est, lh_rew_pred_1, lh_obs_pred_1, lh_rew_pred_2, lh_obs_pred_2 = loss_histories
    lh_value_td.append(td_L)
    lh_rew_est.append(rew_L)
    lh_rew_pred_1.append(rew_pL)
    lh_obs_pred_1.append(obs_L)
    lh_rew_pred_2.append(rew_pL2)
    lh_obs_pred_2.append(obs_L2)
    
    # Sum
    loss = np.round(td_L + obs_L + rew_L + rew_pL + obs_L2 + rew_pL2, 2)

    string = f"l TD: {td_L}, l R est: {rew_L}, l R pred: {rew_pL}, l O pred: {obs_L}, l R pred2: {rew_pL2}, l O pred2: {obs_L2}, Total: {loss}     "

    return string
    

def plot_train_losses(loss_histories):
    lh_value_td, lh_rew_est, lh_rew_pred_1, lh_obs_pred_1, lh_rew_pred_2, lh_obs_pred_2 = loss_histories
    plt.plot(lh_value_td, label='TD Loss')
    plt.plot(lh_rew_est, label='Reward Loss')
    plt.plot(lh_rew_pred_1, label='1 Reward Pred Loss')
    plt.plot(lh_obs_pred_1, label='1 Pred Obs Loss')
    plt.plot(lh_rew_pred_2, label='2 Reward Pred Loss')
    plt.plot(lh_obs_pred_2, label='2 Pred Obs Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.show()


def save_checkpoint(model, epoch, checkpoint_dir="checkpoints", filename=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"
    path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")