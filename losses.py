import torch
import torch.nn.functional as F

from nn_models import ModelCollection
from episodes import EpisodeCollection

def gaussian_NLL(pred_mean, pred_std, target):
    """
    Compute the negative log-likelihood of target under a diagonal Gaussian with given mean and std.
    pred_mean: [B, T, D]
    pred_std:  [B, T, D]
    target:    [B, T, D]
    Returns:   [B, T] NLL for each time step (sum over D)
    """
    var = pred_std ** 2 + 1e-8
    nll = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((target - pred_mean) ** 2 / var)
    
    return nll.sum(dim =-1)  # sum over D to get [B, T]

# Value TD loss: MSE between V(z_t) and r_{t+1} + γ V(z_{t+1})
def loss_value_td(values, rewards, mask_traj, lengths, gamma=1.0):
    V_t   = values[:, :-1]                  # [B, T-1]  V(z_0) .. V(z_{T-2})
    V_tp1 = values[:, 1:].clone().detach()  # [B, T-1]  V(z_1) .. V(z_{T-1})
    r_tp1 = rewards[:, 1:]                  # [B, T-1]  r_1    .. r_{T-1}
    mask  = mask_traj[:, 1:]               # [B, T-1]

    # Zero bootstrap at terminal state
    for b, l in enumerate(lengths):
        if l >= 2:
            idx = min(l - 2, V_tp1.shape[1] - 1)
            V_tp1[b, idx] = 0.0

    td_target = r_tp1 + gamma * V_tp1
    td_error  = (V_t - td_target) ** 2
    td_error  = td_error * mask

    return td_error.sum() / mask.sum()

def loss_q_td(
    q_values,     # [B, T-1] = Q(z_t, a_t)
    rewards,      # [B, T]
    mask_traj,    # [B, T]
    lengths,
    gamma=1.0
):
    # q_values: [B, T-1] where q_values[:, k] = Q(z_k, a_k)
    #   (already aligned by caller)
    #
    # SARSA-style: target_k = r_{k+1} + γ Q(z_{k+1}, a_{k+1})

    q_t    = q_values[:, :-1]                   # [B, T-2]  Q(z_k, a_k)      k=0..T-3
    q_tp1  = q_values[:, 1:].clone().detach()   # [B, T-2]  Q(z_{k+1}, a_{k+1})
    r_tp1  = rewards[:, 1:-1]                   # [B, T-2]  r_{k+1}          k=0..T-3
    mask   = mask_traj[:, 1:-1]                 # [B, T-2]


    # Zero bootstrap at terminal transitions
    for b, L in enumerate(lengths):
        if L >= 2:
            # Last valid transition is k = L-2 in q_values,
            # which is index L-2 in q_t (since q_t starts at k=0)
            idx = min(L - 2, q_tp1.shape[1] - 1)
            q_tp1[b, idx] = 0.0

    td_target = r_tp1 + gamma * q_tp1
    td_error  = (q_t - td_target) ** 2
    td_error  = td_error * mask

    denom = mask.sum()
    if denom.item() == 0:
        return td_error.mean() * 0.0
    return td_error.sum() / denom


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



def loss_obs(
        est_o_params    : torch.tensor, 
        o               : torch.tensor,
        discrete        : bool, 
        mask            : torch.tensor, 
        pred_steps      : int
        ):
    """
    Cross entropy losses for observation prediction
    For discrete observations, est_o_params are interpreted as logits for a categorical distribution, and o is the one-hot observation.
    For continuous observations, est_o_params are interpreted as concatenated mean and std vectors for a diagonal Gaussian, and o is the continuous observation vector.
    """

    pred_obs_target = o[:, pred_steps:, :]        # o_{t+1}
    aux_mask        = mask[:, pred_steps:]        # mask for t+1

    if discrete:
        # Compute cross-entropy loss
        logits          = est_o_params.transpose(1, 2)
        target_labels   = pred_obs_target.argmax(dim=-1)  # [B, T - pred_steps]
        aux_loss = F.cross_entropy(logits, target_labels, reduction='none')  # [B, T - pred_steps]

    else:
        # Compute Gaussian NLL loss
        pred_mean = est_o_params[:, :, :est_o_params.shape[-1]//2]
        pred_std  = est_o_params[:, :, est_o_params.shape[-1]//2:]
        pred_std = F.softplus(pred_std) + 1e-6  # Ensure std is positive
        aux_loss = gaussian_NLL(pred_mean, pred_std, pred_obs_target)  # [B, T - pred_steps]

    aux_loss = aux_loss * aux_mask  # Apply mask
    return aux_loss.sum() / aux_mask.sum()


def compute_model_loss(
        all_chunks:         list[EpisodeCollection],
        models:             ModelCollection,
        gamma,
        lambda_value=1.0,
        lambda_world=1.0,
        n_chunks_past=10,
        device="cpu"
    ):
    """
    Compute loss that updates:
        - belief_rnn
        - critics (value head, q value head)
        - world model heads (reward_head, pred_head, obs_head)

    Returns:
        model_loss : scalar tensor
        log_dict   : {'value': ..., 'world': ...}
    """

    # Last few chunks from the complete collection, presented in reverse order (first element is most recent)
    chunks_to_process = list(reversed(all_chunks[-n_chunks_past:]))
    n_chunks = len(chunks_to_process)


    pred_loss  = torch.tensor(0.0, device=device)
    critic_loss = torch.tensor(0.0, device=device)

    for i, EC in enumerate(chunks_to_process):
        # Move data to device
        h    = EC.batch_histories.to(device)
        a    = EC.batch_actions.to(device)
        obs  = EC.batch_observations.to(device)
        rew  = EC.batch_rewards.to(device)
        mask = EC.batch_mask_traj.to(device)
        actions = EC.batch_actions.to(device)

        # Calculate hidden state
        z = models.belief_model(h)  # full gradients allowed

        # ---------------------------------------------------
        # VALUE AND Q-VALUE LOSS (CRITIC LOSS)
        # ---------------------------------------------------
        if lambda_value > 0.0:
            V_loss = torch.tensor(0.0, device=device)
            Q_loss = torch.tensor(0.0, device=device)

            # If there are enough value heads
            if i < len(models.v_models):
                V_full   = models.v_models[i](z)                           # [B, T]
                
                # 1) state-value TD loss
                V_loss += loss_value_td(
                    values=V_full, rewards=rew, mask_traj=mask, lengths=EC.ep_lengths, gamma=gamma
                    )

            # If there are enough q heads
            if i < len(models.q_models):
                #3) Q-value TD loss
                Q_full   = models.q_models[i](z[:, :-1, :], a[:, 1:, :])   # [B, T-1]

                Q_loss += loss_q_td(
                    q_values=Q_full, rewards=rew, mask_traj=mask, lengths=EC.ep_lengths, gamma=gamma
                )

            critic_loss += (V_loss + Q_loss)
            if i == 0:  # log losses from the most recent chunk
                log_v_loss = V_loss.item()
                log_q_loss = Q_loss.item()

        # ---------------------------------------------------
        # WORLD MODEL LOSS
        # ---------------------------------------------------
        if lambda_world > 0.0:
            # Reward 0 Step
            L_r0 = loss_reward(models.rew_model(z), rew, mask, pred_steps=0)

            # 1-step latent rollout
            pred_z1 = models.pred_model(z, actions, pred_steps=1)
            # Reward 1 step
            L_r1 = loss_reward(models.rew_model(pred_z1), rew, mask, pred_steps=1)
            # Obs 1 step
            L_o1 = loss_obs(models.obs_model(pred_z1), obs, models.obs_discrete, mask, pred_steps=1)

            # 2-step latent rollout
            pred_z2 = models.pred_model(pred_z1, actions, pred_steps=2)
            # Reward 2 step
            L_r2 = loss_reward(models.rew_model(pred_z2), rew, mask, pred_steps=2)
            # Obs 2 step
            L_o2 = loss_obs(models.obs_model(pred_z2), obs, models.obs_discrete, mask, pred_steps=2)

            # Total 
            pred_loss += (L_r0 + L_r1 + L_o1 + L_r2 + L_o2)

            
               
    # Total loss is from both critic loss and world loss
    critic_loss = (lambda_value * critic_loss) / n_chunks
    pred_loss   = (lambda_world * pred_loss ) / n_chunks
    total_loss = critic_loss + pred_loss

    logs = {
        "value": critic_loss.item(),
        "world": pred_loss.item(),
        "v_loss": log_v_loss if lambda_value > 0.0 else -1,
        "q_loss": log_q_loss if lambda_value > 0.0 else -1,
        "r_loss": L_r0.item() + L_r1.item() + L_r2.item() if lambda_world > 0.0 else -1,
        "o_loss": L_o1.item() + L_o2.item() if lambda_world > 0.0 else -1
    }

    return total_loss, logs



def compute_actor_loss(
        newest_chunk: EpisodeCollection,
        models:       ModelCollection,
        gamma,
        lambda_actor=1.0,
        lambda_value=1.0,
        device="cpu"
    ):
    """
    Compute actor loss
    """

    mask_new = newest_chunk.batch_mask_traj.to(device)
    mask_actor = mask_new[:, 1:]                        # [B, T-1]
    act_new  = newest_chunk.batch_actions.to(device)    # [B, T, A]
    
    # Compute latents with *current* belief_rnn
    h_new    = newest_chunk.batch_histories.to(device)
    z_full   = models.belief_model(h_new).detach()      # [B, T, H]    

    

    # Compute the advantage
    advantage = compute_advantage(                      # [B, T-1]
        newest_chunk=newest_chunk,
        z_full=z_full,
        valid_mask=mask_new,
        act_new=act_new,
        models=models,
        gamma=gamma,
        lambda_value=lambda_value,
        device=device
    )

    # Compute action parameters with *current* actor
    param_actor = models.actor_model.forward(z_full)    # [B, T, P]
    
    # Compute log probabilities of each action taken
    log_probs      = models.actor_model.get_action_log_probs(param_actor[:, :-1, :], act_new[:, 1:, :])  # [B, T-1]
    
    # Compute entropy of the action distribution for regularization
    entropy        = models.actor_model.get_action_entropies(param_actor[:, :-1, :])                     # [B, T-1]

    

    policy_term    = (advantage * log_probs * mask_actor).sum() / mask_actor.sum()
    entropy_term   = (entropy * mask_actor).sum() / mask_actor.sum()
    if models.actions_discrete:
        entropy_coeff  = 0.1  # tune this
    else:
        entropy_coeff  = 0.0 # smaller for continuous actions

    actor_loss     = -lambda_actor * policy_term - entropy_coeff * entropy_term

    return actor_loss, {"actor": actor_loss.item()}


def compute_advantage(
        newest_chunk:   EpisodeCollection,
        z_full:         torch.tensor,
        valid_mask:     torch.tensor,
        act_new:        torch.tensor,
        models:         ModelCollection,
        gamma,
        lambda_value=1.0,   
        device="cpu"
    ):
    # ---------------------------------------------------
    # ACTOR LOSS: use advantages A(z_t, a_t) = Q(z_t, a_t) - V(z_t)
    # ---------------------------------------------------
    if lambda_value > 0:
        # State-value critic and Q-value critic for baseline
        V_t = models.v_models[0](z_full[:, :-1, :])                           # [B, T-1]
        Q_t = models.q_models[0](z_full[:, :-1, :], act_new[:, 1:, :])        # [B, T-1] (first action is a dummy vector)

        advantage = Q_t.detach() - V_t.detach() # [B, T-1]

    # ---------------------------------------------------
    # ACTOR LOSS: use REINFORCE baseline
    # ---------------------------------------------------
    elif lambda_value == 0:
        # Monte Carlo returns as baseline
        mc_returns = newest_chunk.get_monte_carlo_returns(gamma).to(device)  # [B, T]
        
        advantage = mc_returns[:, :-1].detach()                                    # [B, T-1], align with z_0..z_{T-2}

    # Normalise over valid entries (mean-center)
    mask_actor = valid_mask[:, 1:]
    valid_adv  = advantage[mask_actor == 1.0]
    advantage  = (advantage - valid_adv.mean())

    return advantage.detach()