
import torch
from episodes import EpisodeCollection, compute_cw_success, compute_reach_success
from losses import compute_model_loss, compute_actor_loss
from actor import collect_episodes_actor, ActorPolicyWrapper
import torch.nn.utils as nn_utils
from train_perf_logger import TrainLogger
from environment import PomdpEnv, CliffWalk, ReacherEnv
from nn_models import ModelCollection, save_checkpoint



def train_with_chunks(
        env             : PomdpEnv,
        models          : ModelCollection,
        optimizers,
        num_new_chunks=50,
        ep_per_chunk=10,
        gamma=0.95,
        actor_steps=10,
        world_steps=20,
        lambda_actor=1.0,
        lambda_value=1.0,
        lambda_world=1.0,
        n_chunks_past=10,
        device="cuda",
        save_checkp=False
    ):
    
    # Initialise loggers for agent performance and train losses
    logs_agent_perf = TrainLogger()
    logs_train_loss = TrainLogger()

    # Optionally save belief RNN
    if save_checkp:
        save_checkpoint(models.belief_model, 0, checkpoint_dir=f"checkpoints", filename=None)


    # Initialise memory
    memory = []

    # Iterate through new chunks
    for chunk_idx in range(1, num_new_chunks+1):
        print(f"[ Chunk {chunk_idx}]")

        # Generate new chunk and append it to memories
        memory = generate_chunks(
            models=models, 
            memory=memory, 
            env=env, 
            ep_per_chunk=ep_per_chunk,
            device=device
            )
        
        # Track agent performance in the new chunk
        logs_agent_perf = chunk_metrics(
            EC=memory[-1], 
            env=env,
            gamma=gamma,
            logs_agent_perf=logs_agent_perf
            )

        # Optimise world model and agent
        models, world_metrics, agent_metrics = optimisation_step(
            memory=memory,
            models=models,
            optimizers=optimizers,
            gamma=gamma,
            n_chunks_past=n_chunks_past,
            lambda_actor=lambda_actor,
            lambda_value=lambda_value,
            lambda_world=lambda_world,
            world_steps=world_steps,
            actor_steps=actor_steps,
            device=device
            )

        # Track world model performance
        logs_train_loss = optim_metrics(
            world_logs=world_metrics, 
            actor_logs=agent_metrics,
            logs_train_loss = logs_train_loss
            )

        # Optionally save belief RNN
        if (chunk_idx - 1) % 10 == 0:
            if save_checkp:
                save_checkpoint(models.belief_model, chunk_idx, filename=None)

    # Save logs
    logs_agent_perf.save_csv("agent_perf.csv")
    logs_train_loss.save_csv("train_loss.csv")


def generate_chunks(
        models          : ModelCollection,
        memory          : list[EpisodeCollection],
        env             : PomdpEnv,
        ep_per_chunk    : int,
        device
        ):
    """
    Generate a new chunk of episodes from the current nn policy
    """

    # Wrap the nn actor to be compatible with PomdpEnv
    actor_policy = ActorPolicyWrapper(belief_rnn=models.belief_model, actor=models.actor_model, device=device)
    
    # Collect episodes using actor
    eps = collect_episodes_actor(env, actor_policy, num_episodes=ep_per_chunk)
    
    # Add new chunk to memories 
    memory.append(EpisodeCollection(eps))

    return memory



def chunk_metrics(
        EC              : EpisodeCollection,
        env             : PomdpEnv,
        gamma           : float,
        logs_agent_perf : TrainLogger
        ):
    """
    Measure various metrics of agent performance in a chunk, print, and log them. 
    """
    
    with torch.no_grad():
    # Mean and std. of return
        returns_tensor = EC.get_monte_carlo_returns(gamma)
        episode_returns = returns_tensor[:, 0]
        mean_return = episode_returns.mean().item()
        std_return = episode_returns.std().item()
        
    # Mean length    
        mean_length = EC.batch_mask_traj.sum().item() / EC.B

    metrics = {}
    metrics['return_mean']  = float(mean_return)
    metrics['return_std']   = float(std_return)
    metrics['traj_len_mean']= float(mean_length)

    printstr = f"traj_len_mean={metrics['traj_len_mean']:.2f}, return_mean={metrics['return_mean']:.2f}, return_std={metrics['return_std']:.2f}"

    # Calculate environment-specific success metrics
    if type(env) == CliffWalk:
        success_rate = compute_cw_success(EC=EC, env=env)
        metrics['success_rate'] = float(success_rate)
        printstr += f", success_rate={metrics['success_rate']:.2f}"
   
    elif type(env) == ReacherEnv:
        success_rate, avg_dist = compute_reach_success(EC=EC, threshold=0.02)
        metrics['success_rate'] = float(success_rate)
        metrics['avg_dist'] = float(avg_dist)
        printstr += f", success_rate={metrics['success_rate']:.2f}, avg_dist={metrics['avg_dist']:.2f}"

    
    
    
    
    print(printstr)    

    # Append to logs
    logs_agent_perf.append(metrics)

    return logs_agent_perf



def optimisation_step(
        memory          : list[EpisodeCollection],
        models          : ModelCollection,
        optimizers,
        gamma           : float,
        n_chunks_past   : int,
        lambda_actor    : float,
        lambda_value    : float,
        lambda_world    : float,
        world_steps     : int,
        actor_steps     : int,
        device
        ):
    """
    Optimise the world model and actor on one chunk of data.
    """
    
    MAX_GRAD_NORM_MODEL = 2.0
    MAX_GRAD_NORM_ACTOR = 1.0
    
    # Unpack optimisers
    optimizer_model, optimizer_actor = optimizers

    # ============================================================
    # PHASE 1: update world model + critic *first*
    # ============================================================
    if lambda_world > 0 or lambda_value > 0:
        for _ in range(world_steps):
            world_loss, world_logs = compute_model_loss(
                all_chunks=memory,
                models=models,
                gamma=gamma,
                lambda_value=lambda_value,
                lambda_world=lambda_world,
                n_chunks_past=n_chunks_past,
                device=device
            )

            optimizer_model.zero_grad()
            world_loss.backward()
            # ---- gradient clipping for model + critics ----
            nn_utils.clip_grad_norm_(
                [p for group in optimizer_model.param_groups for p in group['params']],
                max_norm=MAX_GRAD_NORM_MODEL
            )
            optimizer_model.step()
    
    else:
        world_loss, world_logs = None, None

    # ============================================================
    # PHASE 2: update actor using *updated* latents & critic
    # ============================================================
    for _ in range(actor_steps):
        actor_loss, actor_logs = compute_actor_loss(
            newest_chunk=memory[-1],
            models=models,
            gamma=gamma,
            lambda_actor=lambda_actor,
            lambda_value=lambda_value,
            device=device
        )
        
        optimizer_actor.zero_grad()
        actor_loss.backward()
        # ---- gradient clipping for actor ----
        nn_utils.clip_grad_norm_(
            [p for group in optimizer_actor.param_groups for p in group['params']],
            max_norm=MAX_GRAD_NORM_ACTOR
        )
        optimizer_actor.step()

    return models, world_logs, actor_logs



def optim_metrics(
        world_logs      : dict | None, 
        actor_logs      : dict, 
        logs_train_loss : TrainLogger
        ):
    metrics = {}
    if world_logs != None:
        metrics['value_loss']   = world_logs['value']
        metrics['pred_loss']    = world_logs['world']
    else:
        metrics['value_loss']   = -1
        metrics['pred_loss']    = -1

    metrics['actor_loss']   = actor_logs['actor']

    # Print
    printstr = ""
    if metrics['value_loss'] != -1:
        printstr += f"value_loss={metrics['value_loss']:.2f} "
        printstr += f"q_loss={world_logs['q_loss']:.2f} "
        printstr += f"v_loss={world_logs['v_loss']:.2f} "
        
    if metrics['pred_loss'] != -1: 
        printstr += f"pred_loss={metrics['pred_loss']:.2f} "
        printstr += f"r_loss={world_logs['r_loss']:.2f} "
        printstr += f"o_loss={world_logs['o_loss']:.2f} "
    
    printstr += f"actor_loss={metrics['actor_loss']:.2f}"

    print(printstr)


    # Append to logs
    logs_train_loss.append(metrics)

    return logs_train_loss
