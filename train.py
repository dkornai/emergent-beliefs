import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from episodes import Episode, EpisodeCollection
from belief_rnn import loss_value_td, loss_value_mc, loss_obs, loss_reward

# Training loop
def train(belief_model      : nn.Module, 
          value_model       : nn.Module, 
          reward_model      : nn.Module,
          pred_model        : nn.Module,
          obs_model         : nn.Module, 
          episodes          : EpisodeCollection, 
          test_episode      : Episode, 
          value=None, 
          num_epochs=10, 
          gamma=1.0, 
          lr=1e-3, 
          validate_every=500, 
          batch_size=32, 
          indices=[0, None], 
          lambda_opl=1.0, 
          lambda_rel=1.0, 
          lambda_rpl=1.0, 
          lambda_mc=1.0,
          plot_validation=True,
          env=None,
          ):
    
    # Set up input and target tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
    
    assert type(episodes) == EpisodeCollection, f"episodes must be an EpisodeCollection object, not {type(episodes)}"
    histories       = episodes.batch_histories.to(device) 
    observations    = episodes.batch_observations.to(device)
    actions         = episodes.batch_actions.to(device)
    belief_states   = episodes.batch_beliefs.to(device)
    rewards         = episodes.batch_rewards.to(device) 
    mask_traj       = episodes.batch_mask_traj.to(device)  
    mask_mc         = episodes.batch_mask_mc.to(device)

    mc_returns  = episodes.get_monte_carlo_returns(gamma).to(device)  
    ep_lengths  = episodes.ep_lengths

    value = torch.tensor(value, dtype=torch.float32).to(device) if value is not None else None

    # Validate dimensions
    assert belief_model.input_dim == histories.shape[-1], f"Belief model input dimension {belief_model.input_dim} does not match history dimension {histories.shape[-1]}"

    # Set up the value estimator model
    belief_model.to(device)
    value_model.to(device)
    reward_model.to(device)
    pred_model.to(device)
    obs_model.to(device)
    
    optimizer = torch.optim.Adam(
        list(belief_model.parameters()) + list(value_model.parameters()) + list(reward_model.parameters()) + list(pred_model.parameters()) + list(obs_model.parameters())
        , lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    belief_model.train()



    loss_td_history = []
    loss_mc_history = []
    loss_obs_history = []
    loss_rew_history = []
    loss_next_rew_history = []
    value_pred_history = []

    for epoch in range(num_epochs):
        # Validate and save the model every `validate_every` epochs
        if (epoch) % validate_every == 0:
            belief_model.to('cpu')
            value_model.to('cpu') 
            reward_model.to('cpu')
            pred_model.to('cpu')
            obs_model.to('cpu')
            
            if plot_validation == True:
                
                validate_values(belief_model, value_model, test_episode, value, indices)
                validate_values(belief_model, value_model, episodes.episodes[0], value, indices)
                
                if lambda_rel > 0.0:
                    validate_rewards(belief_model, reward_model, test_episode, indices, env)
                    validate_rewards(belief_model, reward_model, episodes.episodes[0], indices, env)

                if lambda_rpl > 0.0:
                    validate_pred_rewards(belief_model, pred_model, reward_model, test_episode, indices, env)
                    validate_pred_rewards(belief_model, pred_model, reward_model, episodes.episodes[0], indices, env)

                if lambda_opl > 0.0:
                    validate_observations(belief_model, pred_model, obs_model, test_episode, env) 
                    validate_observations(belief_model, pred_model, obs_model, episodes.episodes[0], env) 

            save_checkpoint(belief_model, epoch)
            
            belief_model.to(device)
            value_model.to(device)
            reward_model.to(device)
            pred_model.to(device)
            obs_model.to(device)
            belief_model.train()
            value_model.train()
            reward_model.train()
            pred_model.train()
            obs_model.train()

        # choose "batch_size" random episodes from the dataset
        chosen_i = np.random.choice(len(episodes), size=min(batch_size, len(episodes)), replace=False)
        batch_lengths = [ep_lengths[i] for i in chosen_i]


        optimizer.zero_grad()
        z = belief_model(histories[chosen_i])  # shape: [B, T]
        
        est_values = value_model(z)

        est_rewards = reward_model(z)

        pred_z = pred_model(z, actions[chosen_i])

        pred_rewards = reward_model(pred_z)

        pred_obs = obs_model(pred_z)

        
        # Reward prediction loss
        td_L = loss_value_td(est_values, rewards[chosen_i], mask_traj[chosen_i], batch_lengths, gamma)
        
        # Calculate Monte Carlo loss
        if lambda_mc > 0.0:
            mc_L = loss_value_mc(est_values, mc_returns[chosen_i], mask_mc[chosen_i]) * lambda_mc
        else:
            mc_L = torch.tensor(0.0)
        
        # Calculate reward estimation loss
        if lambda_rel > 0.0:
            rew_L = loss_reward(est_rewards, rewards[chosen_i], mask_traj[chosen_i]) * lambda_rel
        else:
            rew_L = torch.tensor(0.0)

        # Calculate reward prediction loss
        if lambda_rpl > 0.0:
            rew_pL = loss_reward(pred_rewards, rewards[chosen_i][:,1:], mask_traj[chosen_i][:,1:]) * lambda_rpl
        else:
            rew_pL = torch.tensor(0.0)

        # Calculate observation prediction loss
        if lambda_opl > 0.0:
            obs_L = loss_obs(pred_obs, observations[chosen_i], mask_traj[chosen_i]) * lambda_opl
        else:
            obs_L = torch.tensor(0.0)

        

        loss = td_L + mc_L + rew_L + obs_L + rew_pL
        
        # Backpropagation
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to avoid exploding gradients
        optimizer.step()
        scheduler.step()

        # Calculate value prediction loss 
        if value is not None:
            value_pred_error = value_prediction_error(est_values, belief_states[chosen_i], value, mask_traj[chosen_i])
            value_pred_history.append(value_pred_error)

        td_L = np.round(td_L.item(), 2) 
        mc_L = np.round(mc_L.item(), 2) 
        obs_L = np.round(obs_L.item(), 2) 
        rew_L = np.round(rew_L.item(), 2)
        rew_pL = np.round(rew_pL.item(), 2)

        # Print losses
        loss_td_history.append(td_L)
        loss_mc_history.append(mc_L)
        loss_obs_history.append(obs_L)
        loss_rew_history.append(rew_L)
        loss_next_rew_history.append(rew_pL)

        loss = np.round(td_L + mc_L + obs_L + rew_L + rew_pL, 2)
        print(f"Epoch {epoch+1}, TD Loss: {td_L}, MC Loss: {mc_L}, Pred O Loss: {obs_L}, Est R Loss: {rew_L}, Pred R Loss: {rew_pL} Total: {loss}, Value Loss: {value_pred_error}     ", end= '\r')
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, TD Loss: {td_L}, MC Loss: {mc_L}, Pred O Loss: {obs_L}, Est R Loss: {rew_L}, Pred R Loss: {rew_pL} Total: {loss}, Value Loss: {value_pred_error}     ", end= '\n')
         
    belief_model.to('cpu')  # Move model back to CPU after training

    return loss_td_history, loss_mc_history, loss_obs_history, loss_rew_history, value_pred_history, loss_next_rew_history







    
    
def validate_values(model, value_model, test_episode, value, indices):
    true_values = test_episode.belief_states[:,indices[0]:indices[1]] @ np.array(value.cpu())
    true_values = np.round(true_values, 2)
    
    model.eval()  # switch to eval mode
    value_model.eval()
    with torch.no_grad():
        history = torch.tensor(test_episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]

        z = model(history)  # shape: [1, T]
        predicted_values = value_model(z)
        values = predicted_values.squeeze(0).numpy()  # shape: [T]

    values = np.round(values, 2)

    print("True Values:")
    print(true_values)
    print("Predicted Values:")
    print(values)
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label='True Values', marker='o')
    plt.plot(values, label='Predicted Values', marker='x')
    plt.title("True vs Predicted Values")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def validate_rewards(model, reward_model, test_episode, indices, env):
    # 
    true_expected_rewards = test_episode.belief_states[:,indices[0]:indices[1]] @ env.reward_vec
    true_expected_rewards = np.round(true_expected_rewards, 2)

    true_rewards = test_episode.rewards
    
    model.eval()  # switch to eval mode
    reward_model.eval()
    with torch.no_grad():
        history = torch.tensor(test_episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]

        z = model(history)  # shape: [1, T]
        predicted_rewards = reward_model(z)
        rewards = predicted_rewards.squeeze(0).numpy()  # shape: [T]

    rewards = np.round(rewards, 2)

    print("True Expected Rewards:")
    print(true_expected_rewards)
    print("True Observed rewards")
    print(true_rewards)
    print("Estimated Rewards:")
    print(rewards)
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_expected_rewards, label='Expected Rewards', marker='o')
    plt.plot(true_rewards, label='Observed Rewards', marker='o')
    plt.plot(rewards, label='Estimated Rewards', marker='x')
    plt.title("True vs Estimated Rewards")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

def predict_belief_given_action(episode, env):
    true_belief_states = episode.belief_states
    actions = episode.actions
    
    # Calculate prediction of belief given current belief and action 
    optimal_next_belief_states = []
    for i in range(len(true_belief_states) - 1):
        next_belief_pred = true_belief_states[i] @ env.tp_matrix[np.argmax(actions[i+1])]
        optimal_next_belief_states.append(next_belief_pred)

    return optimal_next_belief_states

def validate_pred_rewards(model, pred_model, reward_model, test_episode, indices, env):

    optimal_next_belief_states = predict_belief_given_action(test_episode, env)
    
    true_expected_rewards = optimal_next_belief_states @ env.reward_vec
    true_expected_rewards = np.round(true_expected_rewards, 2)

    true_rewards = test_episode.rewards[1:]
    
    model.eval()  # switch to eval mode
    pred_model.eval()
    reward_model.eval()
    with torch.no_grad():
        history = torch.tensor(test_episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]
        actions = torch.tensor(test_episode.actions, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, A]

        z = model(history)  # shape: [1, T]
        #print(z)
        pred_z = pred_model(z, actions)
        #print(pred_z)
        predicted_rewards = reward_model(pred_z)
        rewards = predicted_rewards.squeeze(0).numpy()  # shape: [T]

    rewards = np.round(rewards, 2)

    print("True Expected Rewards:")
    print(true_expected_rewards)
    print("True Observed rewards")
    print(true_rewards)
    print("Predicted Rewards:")
    print(rewards)
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_expected_rewards, label='Optimal Predicted Rewards', marker='o')
    plt.plot(true_rewards, label='True Observed Rewards', marker='o')
    plt.plot(rewards, label='Predicted Rewards', marker='x')
    plt.title("True vs Predicted Rewards")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()



def validate_observations(model, pred_model, obs_model, test_episode, env):
    optimal_next_belief_states = predict_belief_given_action(test_episode, env)

    true_expected_observation = optimal_next_belief_states @ env.obs_matrix

    true_observations = test_episode.observations[1:]

    with torch.no_grad():
        history = torch.tensor(test_episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]
        actions = torch.tensor(test_episode.actions, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, A]

        z = model(history)  # shape: [1, T]
        #print(z)
        pred_z = pred_model(z, actions)
        #print(pred_z)
        predicted_obs = obs_model(pred_z)
        predicted_obs = F.softmax(predicted_obs, dim=-1).numpy()


    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(predicted_obs[0], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[0].set_title("Predicted Observations")
    axs[1].imshow(true_expected_observation, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[1].set_title("Expected Observations")
    axs[2].imshow(true_observations, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[2].set_title("True Observations")
    
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




def value_prediction_error(pred_values, belief_states, value, mask):
    # Calculate the true values from belief states and the value function
    true_values = belief_states @ value  # shape: [B, T]

    # Calculate the prediction error
    pred_values = pred_values.squeeze(-1)  # Ensure pred_values is [B, T]
    error = (pred_values - true_values) ** 2  # shape: [B, T]
    error = error * mask  # Apply the mask to the error
    error = error.sum() / mask.sum()  # Average over non-masked values
    error = torch.sqrt(error)  # Take the square root to get RMSE

    return np.round(error.item(),2)  # Return the mean squared error as a scalar