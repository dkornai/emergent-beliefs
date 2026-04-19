import torch
import torch.nn.functional as F

from nn_models import ModelCollection
from episodes import EpisodeCollection

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
                V_full   = models.v_models[i](z)                       # [B, T]
                
                # 1) state-value TD loss
                V_loss += loss_value_td(
                    values=V_full, rewards=rew, mask_traj=mask, lengths=EC.ep_lengths, gamma=gamma
                    )

            # If there are enough q heads
            if i < len(models.q_models):
                #3) Q-value TD loss
                Q_full   = models.q_models[i](z)                       # [B, T, A]

                Q_loss += loss_q_td(
                    q_values=Q_full, rewards=rew, actions=actions, mask_traj=mask, lengths=EC.ep_lengths, gamma=gamma
                )

            critic_loss += (V_loss + Q_loss)

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
            L_o1 = loss_obs(models.obs_model(pred_z1), obs, mask, pred_steps=1)

            # 2-step latent rollout
            pred_z2 = models.pred_model(pred_z1, actions, pred_steps=2)
            # Reward 2 step
            L_r2 = loss_reward(models.rew_model(pred_z2), rew, mask, pred_steps=2)
            # Obs 2 step
            L_o2 = loss_obs(models.obs_model(pred_z2), obs, mask, pred_steps=2)

            # Total 
            pred_loss += (L_r0 + L_r1 + L_o1 + L_r2 + L_o2)
               
    # Total loss is from both critic loss and world loss
    critic_loss = (lambda_value * critic_loss) / n_chunks
    pred_loss = (lambda_world * pred_loss ) / n_chunks
    total_loss = critic_loss + pred_loss

    return total_loss, {
        "value": critic_loss.item(),
        "world": pred_loss.item(),
    }



def compute_actor_loss(
        newest_chunk: EpisodeCollection,
        models:       ModelCollection,
        gamma,
        lambda_actor=1.0,
        lambda_value=1.0,
        device="cpu"
    ):
    """
    Compute actor loss ONLY, using the current (already-updated)
    belief_rnn and critic to define advantages.
    """

    # ---------------------------------------------------
    # 1. Latents + basic tensors
    # ---------------------------------------------------

    h_new    = newest_chunk.batch_histories.to(device)
    mask_new = newest_chunk.batch_mask_traj.to(device)
    rew_new  = newest_chunk.batch_rewards.to(device)
    act_new  = newest_chunk.batch_actions.to(device)

    # Compute latents with *current* belief_rnn
    z_full   = models.belief_model(h_new)           # [B, T, H]
    z_det    = z_full.detach()             # don't let actor update the encoder



    

    z_actor        = z_det[:, :-1, :]        # [B, T-1, H], states z_0..z_{T-2}
    actions_actor  = act_new[:, 1:, :]       # [B, T-1, A], actions a_0..a_{T-2}
    mask_actor     = mask_new[:, 1:]         # [B, T-1]
    valid_mask     = (mask_actor == 1.0)

    # ---------------------------------------------------
    # ACTOR LOSS: use advantages A(z_t, a_t) = Q(z_t, a_t) - V(z_t)
    # ---------------------------------------------------
    if lambda_value > 0:
        # state-value critic
        V_full   = models.v_models[0](z_full)                            # [B, T]

        # action-value critic
        Q_full   = models.q_models[0](z_full)                            # [B, T, A]

        # Q(z_t, A_t) for all actions
        Q_t_all        = Q_full[:, :-1, :]                                      # [B, T-1, A]
        Q_t_all        = Q_t_all * mask_actor.unsqueeze(-1)                     # [B, T-1, A]
        
        # Q(z_t, a_t) for the specific action
        Q_sa           = torch.sum(Q_t_all.detach() * actions_actor, dim=-1)    # [B, T-1]

        # V(z_t)
        V_actor        = V_full[:, :-1].detach()  # [B, T-1]
        V_actor        = V_actor * mask_actor

        # Advantage
        A_actor        = Q_sa - V_actor          # [B, T-1]
        A_actor        = A_actor * mask_actor

        # Normalise advantages over valid entries
        valid_adv      = A_actor[valid_mask]
        adv_mean       = valid_adv.mean()
        #adv_std        = valid_adv.std() + 1e-8
        baseline       = (A_actor - adv_mean)# / adv_std

    # ---------------------------------------------------
    # ACTOR LOSS: use REINFORCE baseline
    # ---------------------------------------------------
    elif lambda_value == 0:
        # --- Monte Carlo returns instead of Q − V ---
        mc_returns = newest_chunk.get_monte_carlo_returns(gamma).to(device)  # [B, T]
        G_t = mc_returns[:, :-1].detach()      # [B, T-1], align with z_0..z_{T-2}

        # Normalise over valid entries (mean-center, optionally divide by std)
        valid_G    = G_t[valid_mask]
        baseline   = (G_t - valid_G.mean())    # no baseline, just center


    # Policy at those states
    probs_actor    = models.actor_model(z_actor)          # [B, T-1, A]
    log_probs_all  = torch.log(probs_actor + 1e-8)
    log_probs      = torch.sum(actions_actor * log_probs_all, dim=-1)  # [B, T-1]

    # Entropy bonus
    entropy        = -(probs_actor * log_probs_all).sum(dim=-1)  # [B, T-1]

    policy_term    = (baseline * log_probs * mask_actor).sum() / mask_actor.sum()
    entropy_term   = (entropy * mask_actor).sum() / mask_actor.sum()
    entropy_coeff  = 0.1  # tune this

    actor_loss     = -lambda_actor * policy_term - entropy_coeff * entropy_term

    return actor_loss, {"actor": actor_loss.item()}