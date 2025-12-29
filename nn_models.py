import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical




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
    


class ModelCollection(nn.Module):
    """
    Class to hold a collection of models associated with solving the POMDP
    """

    def __init__(
            self,
            latent_dim, 
            n_actions, 
            n_obs, 
            n_value_models, 
            n_q_models
        ):
        super().__init__()

        # ============================================================
        # Initialise Component Models
        # ============================================================
        self.belief_model = BeliefRNN(input_dim=(n_actions+n_obs), latent_dim=latent_dim)
        self.pred_model   = NextLatentPredictor(latent_dim=latent_dim)
        self.v_models     = nn.ModuleList([ValueReadout(latent_dim=latent_dim) for _ in range(n_value_models)])
        self.q_models     = nn.ModuleList([QReadout(latent_dim=latent_dim, num_actions=n_actions) for _ in range(n_q_models)])
        self.rew_model    = RewardReadout(latent_dim=latent_dim)
        self.obs_model    = ObsReadout(latent_dim=latent_dim, obs_dim=n_obs)
        self.actor_model  = ActorReadout(latent_dim=latent_dim, num_actions=n_actions)

    def init_optimizers(self):
        # Actor optimizer
        optimizer_actor = torch.optim.Adam(self.actor_model.parameters(), lr=1e-4)

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