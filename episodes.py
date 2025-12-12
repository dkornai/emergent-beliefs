import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from environment import PomdpEnv

class Episode():
    """
    Class for handling data from a POMDP episode
    """
    def __init__(self):
        self.states = []
        self.observations = []
        self.rewards = []
        self.actions = []
        self.belief_states = []

        self.attached = False
        
    def add_step(self, state, observation, reward, action, belief_state):
        """
        Add a timestep to the episode data

        :param state:           Current state of the environment, one-hot encoded numpy vector
        :param observation:     Current observation of the environment, one-hot encoded numpy vector
        :param reward:          Reward received from the environment, scalar
        :param action:          Action taken in the environment, one-hot encoded numpy vector
        :param belief_state:    Current belief state, vector of probabilities
        """
        self.states.append(state)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(action)
        self.belief_states.append(belief_state)

    def finish_episode(self):
        """
        Finish the episode and attach the data
        """
        self.attach()

    def attach(self):
        """
        Turn the lists into numpy arrays for easier handling
        """
        if not self.attached:
            self.states         = np.array(self.states)
            self.observations   = np.array(self.observations)
            self.rewards        = np.array(self.rewards)    
            self.actions        = np.array(self.actions)
        
            # Create a history from column-concatenating the observations and actions
            self.history        = np.concatenate((self.observations, self.actions), axis=1)
            self.belief_states  = np.array(self.belief_states)
           
            self.attached = True

    def render(self):
        """
        Render the episode data
        """
        self.attach()
        
        fig, axs = plt.subplots(5, 1, figsize=(8, 10))
        
        # Render the states
        axs[0].imshow(self.states.T, aspect='auto', cmap='gray')
        axs[0].set_title("States")
        axs[0].set_xlabel("Time Step")
        axs[0].set_ylabel("State Index")

        # Render the observations
        axs[1].imshow(self.observations.T, aspect='auto', cmap='gray')
        axs[1].set_title("Observations")
        axs[1].set_xlabel("Time Step")
        axs[1].set_ylabel("Observation Index")

        # Render the actions
        axs[2].imshow(self.actions.T, aspect='auto', cmap='gray')
        axs[2].set_title("Actions")
        axs[2].set_xlabel("Time Step")
        axs[2].set_ylabel("Action Index")

        # Render the rewards
        axs[3].plot(self.rewards, marker='o', linestyle='-', color='b')
        axs[3].set_title("Rewards")
        axs[3].set_xlabel("Time Step")

        # Render the belief states
        axs[4].imshow(self.belief_states.T, aspect='auto', cmap='gray')
        axs[4].set_title("Belief States")
        axs[4].set_xlabel("Time Step")
        axs[4].set_ylabel("State Index")
        
        plt.tight_layout()
        plt.show()




class EpisodeCollection():
    """
    Class for handling data from a collection of episodes. 
    Also supports the calculation of various quantities.
    """
    def __init__(self, episodes):
        
        assert isinstance(episodes, list), "Episodes must be provided as a list."
        assert len(episodes) > 0, "The list of episodes cannot be empty."
        assert all(isinstance(ep, Episode) for ep in episodes), "All items in the list must be Episode objects."
        assert all(ep.attached for ep in episodes), "All episodes must be attached (data converted to numpy arrays)."
        self.episodes = episodes

        # Pre-convert episodes to padded tensors for training
        self.episodes_to_batch()
        
        self.B = len(self.episodes)                         # Number of episodes
        self.T = self.batch_histories.shape[1]              # Length of the longest episode in the batch
        self.O = self.episodes[0].observations.shape[1]     # Observation dimension
        self.A = self.episodes[0].actions.shape[1]          # Number of distinct actions
        self.H = self.batch_histories.shape[2]              # History dimension (O + A)
        self.S = self.batch_beliefs.shape[2]                # Number of states in the belief state (and the environment)

        

    def get_monte_carlo_state_values(self, gamma=None):
        """
        Monte Carlo estimation of V(s) from the list of episodes
        """
        assert gamma is not None, "Gamma (discount factor) must be provided."
        assert gamma >= 0 and gamma <= 1, "Gamma must be in the range [0, 1]."
        assert len(self.episodes) > 100, "Not enough episodes provided for Monte Carlo estimation."
        
        num_states = self.episodes[0].states.shape[1]
        state_returns = np.zeros(num_states)
        state_counts = np.zeros(num_states)

        for ep in self.episodes:
            rewards = ep.rewards
            states = ep.states

            G = 0.0
            returns = [0.0] * len(rewards)
            for t in reversed(range(len(rewards))):
                G = rewards[t] + gamma * G
                returns[t] = G

            for t, state in enumerate(states):
                state_idx = np.argmax(state).item()  # Get index from one-hot
                state_returns[state_idx] += returns[t]
                state_counts[state_idx] += 1

        # Avoid divide-by-zero
        nonzero_mask = state_counts > 0
        state_values = np.zeros(num_states)
        state_values[nonzero_mask] = state_returns[nonzero_mask] / state_counts[nonzero_mask]

        self.mc_values = np.round(state_values, 2)
        return self.mc_values
    
    def get_monte_carlo_returns(self, gamma=None):
        """
        Calculate the Monte Carlo returns for each timestep in each episode.
        """
        assert gamma is not None, "Gamma (discount factor) must be provided."
        assert gamma >= 0 and gamma <= 1, "Gamma must be in the range [0, 1]."

        returns = torch.zeros_like(self.batch_rewards)
        for b in range(self.B):
            G = 0.0
            for t in reversed(range(self.T)):
                if self.batch_mask_traj[b, t] == 1.0:
                    G = self.batch_rewards[b, t] + gamma * G
                    returns[b, t] = G
        
        self.mc_returns = returns
        return self.mc_returns

    def get_history_values(self, gamma=None):
        """
        Compute Monte Carlo value estimates for all unique histories.
        
        Returns a table (list of dicts) with:
            - 'history': the history vector
            - 'count': number of occurrences
            - 'mean_return': mean Monte Carlo return following that history
            - 'var_return': variance of returns
        """
        assert gamma is not None, "Gamma must be provided."
        assert 0 <= gamma <= 1, "Gamma must be in [0,1]."

        # Ensure MC returns already computed
        if not hasattr(self, "mc_returns"):
            self.get_monte_carlo_returns(gamma)

        batch_histories_np = self.batch_histories.cpu().numpy()

        history_dict = {}  # key = tuple(history_vector), value = list of returns

        for b in range(self.B):
            length = self.ep_lengths[b]

            # Extract full unpadded history for episode b
            full_hist = batch_histories_np[b, :length]   # [T, H]

            # Build cumulative histories:
            # concat full_hist[0:t+1] for each t
            for t in range(length):
                # Extract prefix history matrix [t+1, H]
                prefix = full_hist[:t+1]

                # Flatten to a 1D vector
                prefix_vec = prefix.flatten()

                # Hashable key (use bytes for speed)
                key = prefix_vec.tobytes()

                # MC return from timestep t
                G = float(self.mc_returns[b, t].item())

                if key not in history_dict:
                    history_dict[key] = {
                        "vec": prefix_vec.copy(),   # store for later reconstruction
                        "returns": []
                    }
                history_dict[key]["returns"].append(G)

        # Build output table
        table = []
        for entry in history_dict.values():
            arr = np.array(entry["returns"])
            table.append({
                "history": entry["vec"],
                "count": len(arr),
                "mean_return": float(arr.mean()),
                "var_return": float(arr.var(ddof=1)) if len(arr) > 1 else 0.0
            })
        
        # Print feedback
        print("History value estimation complete.")
        print(f"{len(table)} unique histories found in dataset.")

        # Build a lookup map for fast queries:
        self.history_mean_lookup = {
            entry["history"].tobytes(): entry["mean_return"]
            for entry in table
        }

        self.history_value_table = table
        return table

    def query_history_value(self, history_prefix):
        """
        Given a partial history (2D array of shape [t+1, H], or flattened 1D vector),
        return its mean Monte Carlo return if it exists in the dataset.
        Otherwise return np.nan.
        """

        # Ensure history values have been computed
        if not hasattr(self, "history_mean_lookup"):
            raise RuntimeError("Call history_values(gamma) before querying.")

        # Convert to 1D vector if needed
        history_prefix = np.asarray(history_prefix)

        if history_prefix.ndim == 2:
            history_prefix = history_prefix.reshape(-1)
        elif history_prefix.ndim != 1:
            raise ValueError("History must be a 1D or 2D numpy array.")

        key = history_prefix.tobytes()

        return self.history_mean_lookup.get(key, np.nan)


    def episodes_to_batch(self):
        """
        Convert a list of episodes into padded tensors of equal length for training.
        """
        # Extract data from episodes
        histories   = [torch.tensor(ep.history, dtype=torch.float32) for ep in self.episodes]
        observations= [torch.tensor(ep.observations, dtype=torch.float32) for ep in self.episodes]
        rewards     = [torch.tensor(ep.rewards, dtype=torch.float32) for ep in self.episodes]
        beliefs     = [torch.tensor(ep.belief_states, dtype=torch.float32) for ep in self.episodes]
        lengths     = [len(r) for r in rewards]
        
        # Pad sequences to create a batch
        padded_histories    = pad_sequence(histories,   batch_first=True) # [B, T, O+A]
        padded_observations = pad_sequence(observations,batch_first=True) # [B, T, O]
        padded_rewards      = pad_sequence(rewards,     batch_first=True) # [B, T]
        padded_beliefs      = pad_sequence(beliefs,     batch_first=True) # [B, T, S]
        padded_actions      = padded_histories[:, :, -4:]                 # [B, T, A]

        # Mask for valid time steps within each episode [size (B, T)]
        mask_traj   = torch.zeros_like(padded_rewards, dtype=torch.float32)
        for i, length in enumerate(lengths):
            mask_traj[i, :length] = 1.0

        # Composite mask for start and terminal states [size (B, T)]
        mask_monte_carlo = torch.zeros_like(padded_rewards, dtype=torch.float32)
        for i, length in enumerate(lengths):
            mask_monte_carlo[i, length-1] = 1.0  
            mask_monte_carlo[i, 0] = 1.0
        mask_monte_carlo = mask_monte_carlo.clamp(0, 1)


        self.ep_lengths         = lengths
        self.batch_histories    = padded_histories
        self.batch_observations = padded_observations
        self.batch_actions      = padded_actions
        self.batch_rewards      = padded_rewards
        self.batch_beliefs      = padded_beliefs
        self.batch_mask_traj    = mask_traj
        self.batch_mask_mc      = mask_monte_carlo
    
    def __len__(self):
        return self.B
        


def collect_episodes(env:PomdpEnv, policy:np.array, num_episodes:int) -> list[Episode]:
    """
    Collect episodes from the environment using a given policy.
    Args:
        env:            The environment to collect episodes from.
        policy:         A callable that takes the current state and returns an action.
        num_episodes:   Number of episodes to collect.
    Returns:
        A list of "Episode" objects.
    """
    assert isinstance(env, PomdpEnv), "Environment must be an instance of PomdpEnv."
    

    episodes = []
    for _ in range(num_episodes):
        # Initialize a new episode
        episode = Episode()

        # Reset the environment, and emit the first data points
        state, observation, reward, belief, done = env.reset()
        prev_action = np.zeros(len(env.action_space))  # No action taken yet
        
        while not done:
            # Add step to the episode
            episode.add_step(state, observation, reward, prev_action, belief)
            
            # Sample state for the policy from the belief state, and act as if in that state
            act_as_if_state = np.random.choice(env.n * env.m, p=belief)
            action = np.random.choice(env.action_space, p=policy[act_as_if_state])
            prev_action = np.zeros(len(env.action_space))
            prev_action[action] = 1
            
            # Step the environment with the chosen action
            state, observation, reward, belief, done = env.step(action)

        # update the episode with the final states, finish, and append it to the list
        episode.add_step(state, observation, reward, prev_action, belief)
        episode.finish_episode()
        episodes.append(episode)
    
    return episodes  
        



