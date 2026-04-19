import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Distribution




class BeliefRNN(nn.Module):
    """
    Process the history of observations and actions into a hidden state z_t = g(h_t)
    """
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
    """
    Read out the hidden state to a value prediction V(z_t)
    """
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



# class LegacyQReadout(nn.Module):
#     """
#     Map latent z_t to Q-values Q(z_t, a) for all actions.
    
#     Input:
#         z : [B, T, latent_dim] or [B, latent_dim]
#     Output:
#         Q : [B, T, A] or [B, A]  (broadcast over extra dims)
#     """
#     def __init__(self, latent_dim, num_actions, hidden=128):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.num_actions = num_actions

#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, num_actions)
#         )

#     def forward(self, z):
#         # Works for [B, T, H] or [B, H] transparently
#         return self.net(z)  # last dim becomes num_actions

#     def q_for_actions(self, z, actions_one_hot):
#         """
#         Convenience helper:
#             z              : [B, T, latent_dim]
#             actions_one_hot: [B, T, A] (one-hot)
#         Returns:
#             Q(z_t, a_t)    : [B, T]
#         """
#         q_all = self.forward(z)              # [B, T, A]
#         q_sa  = torch.sum(q_all * actions_one_hot, dim=-1)
#         return q_sa

class QReadout(nn.Module):
    """
    Map latent z_t to Q-values Q(z_t, a_t) for the specified actions a_t.
    
    Input:
        z : [B, T, latent_dim] or [B, latent_dim]
        a : [B, T, A]
    Output:
        Q : [B, T] or [B]  (broadcast over extra dims)
    """
    def __init__(self, latent_dim, num_actions, hidden=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(latent_dim+num_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z, a):
        # Works for [B, T, H+A] or [B, H+A] transparently
        combined_input = torch.cat([z, a], dim=-1)  # [B, T, H+A]
        return self.net(combined_input).squeeze(-1) # last dim becomes 1, squeeze to get [B, T] or [B]


class RewardReadout(nn.Module):
    """
    Linear prediction of reward given the latent state z_t: R(z_t)
    """
    
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 1)

    def forward(self, z):
        x = self.fc1(z)
        
        return x.squeeze(-1)  # [B, T]



class NextLatentPredictor(nn.Module):
    """
    Predict the next latent state z_{t+1} given the current z_t and a_t
    """
    def __init__(self, latent_dim, action_dim, hidden=64):
        super().__init__()
        # non-linear transformation mapping b -> w
        self.net = nn.Sequential(
            nn.Linear(latent_dim+action_dim, hidden),
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
    Parameterize the observation model p(o_t | z_t) as a linear readout from the latent state.
    Returned values are interpreted as logits for a categorical distribution over discrete observations.
    or as the concatenation of the mean and the standard deviation vector for a diagonal Gaussian distribution over continuous observations.
    """
    def __init__(self, latent_dim, obs_dim, obs_discrete):
        super().__init__()
        
        assert isinstance(obs_discrete, bool), "obs_discrete should be a boolean indicating the type of observation space."
        self.discrete = obs_discrete

        if self.discrete:
            self.fc1 = nn.Linear(latent_dim, obs_dim)  # logits for categorical distribution
        else:
            self.fc1 = nn.Linear(latent_dim, obs_dim * 2)  # mean and std for Gaussian distribution

    def forward(self, z):
        x = self.fc1(z)
        
        return x




class ActorReadout(nn.Module):
    """
    Readout from the latent state to a distribution over actions. \pi(a_t | z_t)
    """
    def __init__(self, latent_dim, num_actions, actions_discrete, hidden_dim=128):
        super().__init__()
        self.num_actions = num_actions

        assert isinstance(actions_discrete, bool), "actions_discrete should be a boolean indicating the type of action space."
        self.discrete = actions_discrete

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        if self.discrete:
            self.fc2 = nn.Linear(hidden_dim, num_actions)
        else:
            self.fc2 = nn.Linear(hidden_dim, num_actions * 2)  # mean and std for each action dimension
        
    def forward(self, z):
        # Forward pass through the network to get parameters of the action distribution
        param = self.fc2(F.relu(self.fc1(z)))

        return param
    
    def action_distribution(self, param) -> Distribution:
        """
        Args:
            param: Tensor of shape [B, T, P], where P is the number of parameters output by the network (either num_actions for discrete, or num_actions*2 for continuous)

        Returns:
            dist: A torch Distribution object representing \pi(a_t | z_t)
        """
        if self.discrete:
            # For discrete actions, output is interpreted as logits for a categorical distribution
            dist = Categorical(logits=param)
        else:
            # For continuous actions, output is interpreted as mean and std for a Gaussian distribution
            half = param.shape[-1] // 2
            mean = param[..., :half]
            mean = torch.tanh(mean)  # Optional: constrain mean to a certain range (e.g., [-1, 1]) depending on action space
            # log_std = param[..., half:] + 1e-6  # Ensure std is positive
            # std = torch.exp(log_std)
            # std = torch.clamp(std, min=0.05, max=1.0)  # Optional: clamp std to a reasonable range to prevent numerical issues
            std = 0.1 * torch.ones_like(mean)  # Fixed std for stability, can be learned or output by the network if desired

            cov = torch.diag_embed(std**2)
            dist = MultivariateNormal(mean, cov)

        return dist

    def sample_action(self, z_t) -> np.ndarray:
        """
        Return sampled a_t given z_t, as a numpy array (one-hot for discrete, raw values for continuous)

        Args:
            z_t: Tensor of shape [latent_dim], describing the current latent state of the RNN

        Returns:
            action: numpy array of shape [num_actions], either one-hot (discrete) or raw values (continuous)
        """
        # Get distribution parameters from the network
        param = self.forward(z_t)
        # Set up the action distribution
        dist = self.action_distribution(param)
        # Sample an action
        action = dist.sample()
        
        # Return action in the appropriate format
        if self.discrete:
            # Convert to one-hot encoding for discrete actions
            action_oh = np.zeros(self.num_actions)
            action_oh[action] = 1.0
            return action_oh
        
        else:
            # Return raw values for continuous actions
            return action.squeeze(0).cpu().numpy()

    def get_action_log_probs(self, param, actions):
        """
        Compute log probabilities of given actions.

        Args:
            param:   Tensor of shape [B, T, P], parameters of the action distribution output by the network
            actions: Tensor of shape [B, T, A], with continuous actions or one-hot encoded discrete actions

        Returns:
            log_probs: Tensor of shape [B, T]
        """
        dist = self.action_distribution(param)  # batch shape [B, T]

        if self.discrete:
            # Categorical expects integer class indices
            actions = torch.argmax(actions, dim=-1)  # Convert one-hot to class indices
            log_probs = dist.log_prob(actions)  # [B, T]
        else:
            # MultivariateNormal expects matching shape [B, T, A]
            log_probs = dist.log_prob(actions)  # [B, T]

        return log_probs
    
    def get_action_entropies(self, param):
        """
        Compute entropies of the action distributions.

        Args:
            param: Tensor of shape [B, T, P], parameters of the action distribution output by the network

        Returns:
            entropies: Tensor of shape [B, T]
        """
        dist = self.action_distribution(param)  # batch shape [B, T]
        entropies = dist.entropy()  # [B, T]

        return entropies

        
        

class ModelCollection(nn.Module):
    """
    Class to hold a collection of models associated with solving the POMDP
    """

    def __init__(
            self,
            latent_dim, 
            dim_actions,
            actions_discrete, 
            dim_obs, 
            obs_discrete,
            n_value_models, 
            n_q_models
        ):
        super().__init__()

        self.actions_discrete = actions_discrete
        self.obs_discrete = obs_discrete

        # ============================================================
        # Initialise Component Models
        # ============================================================
        self.belief_model = BeliefRNN(input_dim=(dim_actions+dim_obs), latent_dim=latent_dim)
        self.pred_model   = NextLatentPredictor(latent_dim=latent_dim, action_dim=dim_actions)
        self.v_models     = nn.ModuleList([ValueReadout(latent_dim=latent_dim) for _ in range(n_value_models)])
        self.q_models     = nn.ModuleList([QReadout(latent_dim=latent_dim, num_actions=dim_actions) for _ in range(n_q_models)])
        self.rew_model    = RewardReadout(latent_dim=latent_dim)
        self.obs_model    = ObsReadout(latent_dim=latent_dim, obs_dim=dim_obs, obs_discrete=obs_discrete)
        self.actor_model  = ActorReadout(latent_dim=latent_dim, num_actions=dim_actions, actions_discrete=actions_discrete)

    def init_optimizers(self):
        # Actor optimizer
        if self.actions_discrete:
            optimizer_actor = torch.optim.Adam(self.actor_model.parameters(), lr=1e-4)
        else:
            optimizer_actor = torch.optim.Adam(self.actor_model.parameters(), lr=1e-4)  # higher LR for continuous actions

        # World model optimizer
        core_params   = list(self.belief_model.parameters()) + list(self.pred_model.parameters())
        critic_params = list(self.v_models.parameters())     + list(self.q_models.parameters())
        observ_params = list(self.rew_model.parameters())    + list(self.obs_model.parameters())

        optimizer_model = torch.optim.Adam(
            [
                {"params": core_params,   "lr": 1e-4},
                {"params": critic_params, "lr": 1e-4},  
                {"params": observ_params, "lr": 1e-3},
            ]
        )

        return optimizer_model, optimizer_actor


def save_checkpoint(model, epoch, checkpoint_dir='checkpoints', filename=None):
    """
    Save the parameters of a given model to the disk for later re-use.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"
    path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")