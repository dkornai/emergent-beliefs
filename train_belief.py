import os

import numpy as np
import torch
import pandas as pd

from environment import CliffWalk, cw_default_params_dict
from episodes import collect_episodes, EpisodeCollection
from belief_decoders import NonLinBeliefDecoder, LinBeliefDecoder, decode_training, decode_visualisation, estimate_entropy
from nn_models import BeliefRNN

NUM_EPOCHS = 5000

def belief_test(config):
    # ---------------------------------------------------
    # Init POMDP env, and collect data
    # ---------------------------------------------------
    cliff = CliffWalk(**cw_default_params_dict)
    policy = cliff.get_optimal_policy(epsilon=0.3)

    # Collect episodes from the environment using the target policy
    episode_list = collect_episodes(cliff, policy, num_episodes=5000)

    episodes = EpisodeCollection(episode_list)
    print("collected test episodes")

    # ---------------------------------------------------
    # Calculate Belief Entropy
    # ---------------------------------------------------
    all_belief_states = []
    for episode in episodes.episodes:
        all_belief_states.append(np.array(episode.belief_states))
    all_belief_states = np.concatenate(all_belief_states, axis=0)

    belief_entropy = estimate_entropy(all_belief_states, base=np.e)

    print(f"Average Entropy of Belief over states: {belief_entropy}:")


    # ---------------------------------------------------
    # Calculate decodability from current obs and action
    # ---------------------------------------------------
    indices = []
    indices.append("1step")

    print("Linear Decoder")
    belief_decoder = LinBeliefDecoder(input_dim=episodes.H, belief_dim=episodes.S)
       
    # Train the belief decoder to decode the true belief states from the obs+action history, and calculate the CE and TV losses
    belief_decoder, ce_loss, tv_loss = decode_training(
        episodes, belief_decoder, [0, None], 
        value_RNN=None, num_epochs=NUM_EPOCHS, lr=1e-3)
        
    CEb_true_lin = ce_loss
    TVb_true_lin = tv_loss
    
    
    # ---------------------------------------------------
    # Calculate sufficency from learned recurrent state
    # ---------------------------------------------------
    belief_model = BeliefRNN(input_dim=(cliff.action_dim+cliff.obs_dim), latent_dim=config.RNN_HIDDEN)

    CEs_true_lin = []
    TVs_true_lin = []
    
        
    # Iterate over saved models
    for chunk in config.SAVE_PARAM:
        candidate_filename = f"checkpoints/checkpoint_epoch_{chunk}.pth"
        if os.path.exists(candidate_filename):
            indices.append(chunk)

            # Load the value RNN model from the checkpoint
            model_dict = torch.load(candidate_filename, weights_only=True)
            belief_model.load_state_dict(model_dict['model_state_dict'])
            belief_model.to('cpu')
            print(f"\nUsing Belief RNN from chunk {chunk}")
            
            print("Linear Decoder")
            belief_decoder = LinBeliefDecoder(input_dim=belief_model.latent_dim, belief_dim=episodes.S)

            # Train the belief decoder to decode the true belief states from the RNN's latent states, and calculate the CE and TV losses
            belief_decoder, ce_loss, tv_loss = decode_training(
                episodes, belief_decoder, [0, None], 
                value_RNN=belief_model, num_epochs=NUM_EPOCHS, lr=1e-3)

            CEs_true_lin.append(ce_loss)
            TVs_true_lin.append(tv_loss)



    # Subtract the belief entropy to get the KL divergence
    KL_start_lin = CEb_true_lin - belief_entropy
    KL_train_lin    = np.array(CEs_true_lin) - belief_entropy
    KL_train_lin    = KL_train_lin.tolist()

    # Build dataframe with KL results and TV results
    df = pd.DataFrame({
        "index": indices,
        "kl_linear": [KL_start_lin] + KL_train_lin,
        "tv_linear": [TVb_true_lin] + TVs_true_lin,
    })

    # Save to CSV
    print(df)
    df.to_csv("divergence_results.csv", index=False)  
    print("\nSaved divergence results to divergence_results.csv")
