import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from environment import PomdpEnv
from episodes import Episode, HistoryValueTable

def plot_validate(
        models          : list[nn.Module],
        test_episodes   : list[Episode],
        environment     : PomdpEnv,
        history_values  : HistoryValueTable,
        state_values
        ):
    # Unpack models
    belief_model, value_model, reward_model, pred_model, obs_model = models

    device = next(belief_model.parameters()).device

    belief_model.to('cpu')
    value_model.to('cpu') 
    reward_model.to('cpu')
    pred_model.to('cpu')
    obs_model.to('cpu')
    
    belief_model.eval()
    value_model.eval()
    reward_model.eval()
    pred_model.eval()
    obs_model.eval()
    
    for episode in test_episodes:
        validate_values(belief_model, value_model, episode, history_values, state_values)

        validate_pred_rewards(belief_model, pred_model, reward_model, episode, environment, steps = 0)
        validate_pred_rewards(belief_model, pred_model, reward_model, episode, environment, steps = 1)
        validate_pred_rewards(belief_model, pred_model, reward_model, episode, environment, steps = 2)

        validate_pred_observations(belief_model, pred_model, obs_model, episode, environment, steps = 1) 
        validate_pred_observations(belief_model, pred_model, obs_model, episode, environment, steps = 2) 
    
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




    
def validate_values(belief_model, value_model, test_episode, history_values, state_value):
    true_values = test_episode.belief_states @ np.array(state_value)
    true_values = np.round(true_values, 2)

    if history_values != None:
        true_history_values = []
        for i in range(len(test_episode.history)):
            true_history_values.append(history_values.query(test_episode.history[:i+1]))
    else:
        true_history_values = np.empty(len(test_episode.history))
        true_history_values[:] = np.nan
    
    with torch.no_grad():
        history = torch.tensor(test_episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]

        z = belief_model(history)  # shape: [1, T]
        predicted_values = value_model(z)
        values = predicted_values.squeeze(0).numpy()  # shape: [T]

    values = np.round(values, 2)

    print("Linear Values:")
    print(true_values)
    print("History Values:")
    print(true_history_values)
    print("Predicted Values:")
    print(values)
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label='Linear Values', marker='o')
    plt.plot(true_history_values, label='History Values', marker='o')
    plt.plot(values, label='Predicted Values', marker='x')
    plt.title("True vs Predicted Values")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def predict_belief_given_action(episode, env, steps):
    true_belief_states = episode.belief_states
    actions = episode.actions
    
    if steps == 0:
        return true_belief_states
    
    if steps == 1:

        # Calculate prediction of belief given current belief and action 
        optimal_next_belief_states = []
        for i in range(len(true_belief_states) - 1):
            next_belief_pred = true_belief_states[i] @ env.tp_matrix[np.argmax(actions[i+1])]
            optimal_next_belief_states.append(next_belief_pred)

        return optimal_next_belief_states
    
    if steps == 2:

        # Calculate prediction of belief given current belief and action 
        optimal_next_belief_states = []
        for i in range(len(true_belief_states) - 2):
            next_belief_pred = true_belief_states[i] @ env.tp_matrix[np.argmax(actions[i+1])]
            next_belief_pred = next_belief_pred @ env.tp_matrix[np.argmax(actions[i+2])]
            optimal_next_belief_states.append(next_belief_pred)

        return optimal_next_belief_states



def validate_pred_rewards(belief_model, pred_model, reward_model, test_episode, env, steps):

    optimal_next_belief_states = predict_belief_given_action(test_episode, env, steps)
    
    true_expected_rewards = optimal_next_belief_states @ env.reward_vec
    true_expected_rewards = np.round(true_expected_rewards, 2)

    true_rewards = test_episode.rewards[steps:]
    

    with torch.no_grad():
        history = torch.tensor(test_episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]
        actions = torch.tensor(test_episode.actions, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, A]

        z = belief_model(history)  # shape: [1, T]
        
        if steps in [1, 2]:
            z = pred_model(z, actions, pred_steps=1)
        if steps in [2]:
            z = pred_model(z, actions, pred_steps=2)
        
        predicted_rewards = reward_model(z)
        
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



def validate_pred_observations(belief_model, pred_model, obs_model, test_episode, env, steps):
    optimal_next_belief_states = predict_belief_given_action(test_episode, env, steps)

    true_expected_observation = optimal_next_belief_states @ env.obs_matrix

    true_observations = test_episode.observations[steps:]

    with torch.no_grad():
        history = torch.tensor(test_episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]
        actions = torch.tensor(test_episode.actions, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, A]

        z = belief_model(history)  # shape: [1, T]
        
        if steps in [1, 2]:
            z = pred_model(z, actions, pred_steps=1)
        if steps in [2]:
            z = pred_model(z, actions, pred_steps=2)
        
        predicted_obs = obs_model(z)
        predicted_obs = F.softmax(predicted_obs, dim=-1).numpy()


    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(predicted_obs[0], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[0].set_title("Predicted Observations")
    axs[1].imshow(true_expected_observation, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[1].set_title("Expected Observations")
    axs[2].imshow(true_observations, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axs[2].set_title("True Observations")
    
    plt.show()



