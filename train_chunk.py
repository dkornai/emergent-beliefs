import torch
from episodes import EpisodeCollection
from belief_rnn import loss_value_td, loss_reward, loss_obs, loss_q_td
from actor import compute_returns, MovingBaseline, collect_episodes_actor, ActorPolicyWrapper
from plot_validate import plot_validate

def compute_success_and_cliff_rates(EC: EpisodeCollection, env):
    """
    Computes the fraction of episodes that:
        - reach the goal
        - fall into the cliff
        - neither (unexpected termination or wandering)
    """
    successes = 0
    cliffs = 0
    others = 0

    for ep in EC.episodes:

        final_reward = ep.rewards[-1]

        if final_reward == env.target_reward:
            successes += 1
        elif final_reward == env.cliff_reward:
            cliffs += 1
        else:
            others += 1  # Should be implossible in CliffWalk

    total = len(EC.episodes)

    return {
        "success_rate": successes / total,
        "cliff_rate": cliffs / total,
        "other_rate": others / total
    }



def compute_model_loss(
        newest_chunk: EpisodeCollection,
        all_chunks: list[EpisodeCollection],
        models,
        gamma,
        lambda_value=1.0,
        lambda_world=1.0,
        device="cpu"
    ):
    """
    Compute loss that updates:
        - belief_rnn
        - critic (value head)
        - world model heads (reward_head, pred_head, obs_head)

    Returns:
        model_loss : scalar tensor
        log_dict   : {'value': ..., 'world': ...}
    """
    belief_rnn, actor, critic, q_head, reward_head, pred_head, obs_head = models

    # ----- newest chunk tensors -----
    h_new    = newest_chunk.batch_histories.to(device)
    mask_new = newest_chunk.batch_mask_traj.to(device)
    rew_new  = newest_chunk.batch_rewards.to(device)
    act_new  = newest_chunk.batch_actions.to(device)     # [B, T, A]

    # latent sequence
    z_full   = belief_rnn(h_new)      # [B, T, H]

    # state-value critic
    V_full   = critic(z_full)                            # [B, T]

    # action-value critic
    Q_full   = q_head(z_full)                            # [B, T, A]

    # ---------------------------------------------------
    # CRITIC / Q LOSS (TD) using full z
    # ---------------------------------------------------
    if lambda_value > 0.0:
        # 1) state-value TD loss (as before)
        V_loss = loss_value_td(
            values=V_full,
            rewards=rew_new,
            mask_traj=mask_new,
            lengths=newest_chunk.ep_lengths,
            gamma=gamma
        )

        # 2) Q-value TD loss (new)
        Q_loss = loss_q_td(
            q_values=Q_full,
            rewards=rew_new,
            actions=act_new,
            mask_traj=mask_new,
            lengths=newest_chunk.ep_lengths,
            gamma=gamma
        )

        critic_loss = lambda_value * (V_loss + Q_loss)
        #print(V_loss, Q_loss)
    else:
        critic_loss = torch.tensor(0.0, device=device)

    # ---------------------------------------------------
    # WORLD MODEL LOSS (all chunks) -- unchanged, except it uses belief_rnn only
    # ---------------------------------------------------
    world_loss = torch.tensor(0.0, device=device)

    if lambda_world > 0.0:
        # e.g. just last 10 chunks
        for EC in all_chunks[-10:]:
            h    = EC.batch_histories.to(device)
            obs  = EC.batch_observations.to(device)
            rew  = EC.batch_rewards.to(device)
            mask = EC.batch_mask_traj.to(device)
            actions = EC.batch_actions.to(device)

            z = belief_rnn(h)  # full gradients allowed

            # reward est
            est_rewards = reward_head(z)
            L_r  = loss_reward(est_rewards, rew, mask, pred_steps=0)

            # 1-step
            pred_z1 = pred_head(z, actions, pred_steps=1)
            L_r1 = loss_reward(reward_head(pred_z1), rew, mask, pred_steps=1)
            L_o1 = loss_obs(obs_head(pred_z1), obs, mask, pred_steps=1)

            # 2-step
            pred_z2 = pred_head(pred_z1, actions, pred_steps=2)
            L_r2 = loss_reward(reward_head(pred_z2), rew, mask, pred_steps=2)
            L_o2 = loss_obs(obs_head(pred_z2), obs, mask, pred_steps=2)

            world_loss += (L_r + L_r1 + L_o1 + L_r2 + L_o2)

        world_loss = lambda_world * (world_loss / min(10, len(all_chunks)))

    # ---------------------------------------------------
    # MODEL LOSS = critic + world   (updates belief_rnn + critic + Q + world)
    # ---------------------------------------------------
    model_loss = critic_loss + world_loss

    return model_loss, {
        "value": critic_loss.item(),
        "world": world_loss.item(),
    }

def compute_actor_loss(
        newest_chunk: EpisodeCollection,
        models,
        gamma,
        lambda_actor=1.0,
        device="cpu"
    ):
    """
    Compute actor loss ONLY, using the current (already-updated)
    belief_rnn and critic to define advantages.
    """
    belief_rnn, actor, critic, q_head, reward_head, pred_head, obs_head = models

    # ---------------------------------------------------
    # 1. Latents + basic tensors
    # ---------------------------------------------------

    h_new    = newest_chunk.batch_histories.to(device)
    mask_new = newest_chunk.batch_mask_traj.to(device)
    rew_new  = newest_chunk.batch_rewards.to(device)
    act_new  = newest_chunk.batch_actions.to(device)

    # Compute latents with *current* belief_rnn
    z_full   = belief_rnn(h_new)           # [B, T, H]
    z_det    = z_full.detach()             # don't let actor update the encoder



    # state-value critic
    V_full   = critic(z_full)                            # [B, T]

    # action-value critic
    Q_full   = q_head(z_full)                            # [B, T, A]

    # ---------------------------------------------------
    # ACTOR LOSS: use advantages A(z_t, a_t) = Q(z_t, a_t) - V(z_t)
    # ---------------------------------------------------

    #         # We want transitions at (z_t, a_t). In your logging:
    #   z_t      is at index t
    #   a_t      is stored at actions[:, t+1]
    #   mask_new[:, t] == 1 for valid states
    #
    # So align as:
    #   z_actor       = z_t       for t = 0..T-2  (indices 0..T-2)
    #   actions_actor = a_t       stored at index t+1  -> slice 1..T-1
    #   mask_actor    = mask for those actions   -> slice 1..T-1
    B, T = rew_new.shape

    z_actor        = z_det[:, :-1, :]        # [B, T-1, H], states z_0..z_{T-2}
    actions_actor  = act_new[:, 1:, :]       # [B, T-1, A], actions a_0..a_{T-2}
    mask_actor     = mask_new[:, 1:]         # [B, T-1]

    # Policy at those states
    probs_actor    = actor(z_actor)          # [B, T-1, A]
    log_probs_all  = torch.log(probs_actor + 1e-8)
    log_probs      = torch.sum(actions_actor * log_probs_all, dim=-1)  # [B, T-1]

    # Q(z_t, a_t): use z_t and a_t
    Q_t_all        = Q_full[:, :-1, :]       # [B, T-1, A]
    Q_t_all        = Q_t_all * mask_actor.unsqueeze(-1)
    # print('Q function')
    # print(Q_t_all[0,:,:])
    Q_sa           = torch.sum(Q_t_all.detach() * actions_actor, dim=-1)  # [B, T-1]

    # Baseline V(z_t)
    
    V_actor        = V_full[:, :-1].detach()  # [B, T-1]
    V_actor        = V_actor * mask_actor
    # print('V function')
    # print(V_actor[0,:].reshape(-1,1))

    A_actor        = Q_sa - V_actor          # [B, T-1]
    A_actor        = A_actor * mask_actor
    # print('Adv function')
    # print(A_actor[0,:].reshape(-1,1))


    # Normalise advantages over valid entries
    valid_mask     = (mask_actor == 1.0)
    valid_adv      = A_actor[valid_mask]
    adv_mean       = valid_adv.mean()
    #adv_std        = valid_adv.std() + 1e-8
    A_norm         = (A_actor - adv_mean)# / adv_std

    # entropy bonus
    entropy        = -(probs_actor * log_probs_all).sum(dim=-1)  # [B, T-1]

    policy_term    = (A_norm * log_probs * mask_actor).sum() / mask_actor.sum()
    entropy_term   = (entropy * mask_actor).sum() / mask_actor.sum()
    entropy_coeff  = 0.01  # tune this

    actor_loss     = -lambda_actor * policy_term - entropy_coeff * entropy_term

    return actor_loss, {"actor": actor_loss.item()}


def train_with_chunks(
        env,
        models,
        initial_chunks,
        num_new_chunks=50,
        episodes_per_chunk=10,
        gamma=0.95,
        actor_steps=10,
        world_steps=20,
        lambda_actor=1.0,
        lambda_value=1.0,
        lambda_world=1.0,
        device="cuda",
        plot=False,
    ):

    belief_rnn, actor, critic, q_head, reward_head, pred_head, obs_head = models

    # ------------------------
    # Separate optimizers
    # ------------------------
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_model = torch.optim.Adam(
        list(belief_rnn.parameters()) +
        list(critic.parameters()) +
        list(reward_head.parameters()) +
        list(q_head.parameters()) +
        list(pred_head.parameters()) +
        list(obs_head.parameters()),
        lr=1e-3
    )

    memory = initial_chunks[:]

    for chunk_idx in range(num_new_chunks):

        # Collect episodes on-policy with actor
        actor_policy = ActorPolicyWrapper(belief_rnn, actor, device=device)
        eps = collect_episodes_actor(env, actor_policy, num_episodes=episodes_per_chunk)

        if plot:
            plot_validate(models, eps[0:1], env, None, None)

        EC_new = EpisodeCollection(eps)
        memory.append(EC_new)
        

        # ----------------------------
        # Logging: mean episodic return, length
        # ----------------------------
        with torch.no_grad():
            returns_tensor = EC_new.get_monte_carlo_returns(gamma)
            episode_returns = returns_tensor[:, 0]
            mean_return = episode_returns.mean().item()
            mean_length = EC_new.batch_mask_traj.sum() / episodes_per_chunk

        metrics = compute_success_and_cliff_rates(EC_new, env)
        print(f"[Chunk {chunk_idx}] success={metrics['success_rate']:.2f}, "
              f"cliff={metrics['cliff_rate']:.2f}, "
              f"mean_return={mean_return:.3f}, mean_len={mean_length:.3f}")

        # ============================================================
        # PHASE 1: update world model + critic *first*
        # ============================================================

        last_model_logs = {"value": 0.0, "world": 0.0}

        for _ in range(world_steps):
            model_loss, model_logs = compute_model_loss(
                newest_chunk=EC_new,
                all_chunks=memory,
                models=models,
                gamma=gamma,
                lambda_value=lambda_value,
                lambda_world=lambda_world,
                device=device
            )
            optimizer_model.zero_grad()
            model_loss.backward()
            optimizer_model.step()

            last_model_logs = model_logs  # keep last for printing

        # ============================================================
        # PHASE 2: update actor using *updated* latents & critic
        # ============================================================
        last_actor_logs = {"actor": 0.0}

        for _ in range(actor_steps):
            actor_loss, actor_logs = compute_actor_loss(
                newest_chunk=EC_new,
                models=models,
                gamma=gamma,
                lambda_actor=lambda_actor,
                device=device
            )
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            last_actor_logs = actor_logs  # last for printing

        print(
            f"actor={last_actor_logs['actor']:.3f}, "
            f"value={last_model_logs['value']:.3f}, world={last_model_logs['world']:.3f}, "
        )

    print("Training complete!")
    return models


