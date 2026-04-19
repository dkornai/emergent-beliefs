import os

import numpy as np
import torch
import pandas as pd

from environment import CliffWalk
from episodes import collect_episodes, EpisodeCollection
from belief_decoders import NonLinBeliefDecoder, LinBeliefDecoder, decode_training, decode_visualisation, estimate_entropy
from nn_models import BeliefRNN

NUM_EPOCHS = 5000

def belief_test(config):
    # ---------------------------------------------------
    # Init POMDP env, and collect data
    # ---------------------------------------------------
    cliff = CliffWalk(n=config.N, m=config.M, self_transition_prob=config.SELF_TRANISTION, gamma=config.SELF_TRANISTION)
    policy = cliff.get_optimal_policy(epsilon=0.3)

    # Collect episodes from the environment using the target policy
    episode_list = collect_episodes(cliff, policy, num_episodes=5000)

    episodes = EpisodeCollection(episode_list)


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

    for type in ['lin', 'nonlin']:
        if type == 'lin':
            print("Linear Decoder")
            belief_decoder = LinBeliefDecoder(input_dim=episodes.H, belief_dim=episodes.S)
        else:
            print("Non-Linear Decoder")
            belief_decoder = NonLinBeliefDecoder(input_dim=episodes.H, hidden_dim=64, belief_dim=episodes.S)

        belief_decoder, ce_loss, tv_loss = decode_training(
            episodes, belief_decoder, [0, None], 
            value_RNN=None, num_epochs=NUM_EPOCHS, lr=1e-3)
        
        if type == 'lin':
            CEb_true_lin = ce_loss
            TVb_true_lin = tv_loss
        else:
            CEb_true_nonlin = ce_loss
            TVb_true_nonlin = tv_loss
        
        #decode_visualisation(test_episode, belief_decoder, indices=[0,None], env_size = (3, 8), value_RNN=None)
    
    
    # ---------------------------------------------------
    # 
    # ---------------------------------------------------

    belief_model = BeliefRNN(input_dim=(cliff.action_dim+cliff.obs_dim), latent_dim=config.RNN_HIDDEN)

    CEs_true_lin = []
    CEs_true_nonlin = []
    TVs_true_lin = []
    TVs_true_nonlin = []
    
        
    # Iterate over epochs
    for chunk in range(0, config.NUM_NEW_CHUNKS + 10):
        candidate_filename = f"checkpoints/checkpoint_epoch_{chunk}.pth"
        if os.path.exists(candidate_filename):
            indices.append(chunk)

            # Load the value RNN model from the checkpoint
            model_dict = torch.load(candidate_filename, weights_only=True)
            belief_model.load_state_dict(model_dict['model_state_dict'])
            belief_model.to('cpu')
            print(f"\nUsing Belief RNN from chunk {chunk}")
            
            # Iterate over linear and non-linear decoders
            for type in ['lin', 'nonlin']:
                if type == 'lin':
                    print("Linear Decoder")
                    belief_decoder = LinBeliefDecoder(input_dim=belief_model.latent_dim, belief_dim=episodes.S)
                else:
                    print("Non-Linear Decoder")
                    belief_decoder = NonLinBeliefDecoder(input_dim=belief_model.latent_dim, hidden_dim=64, belief_dim=episodes.S)

                belief_decoder, ce_loss, tv_loss = decode_training(
                    episodes, belief_decoder, [0, None], 
                    value_RNN=belief_model, num_epochs=NUM_EPOCHS, lr=1e-3)
                
                if type == 'lin':
                    CEs_true_lin.append(ce_loss)
                    TVs_true_lin.append(tv_loss)
                else:
                    CEs_true_nonlin.append(ce_loss)
                    TVs_true_nonlin.append(tv_loss)

                # if epoch == 0 or epoch == EPOCHS[-1]:
                #    decode_visualisation(test_episode, belief_decoder, [0, None], env_size = (3, 8), value_RNN=belief_model)

    
    
    KL_start_lin = CEb_true_lin - belief_entropy
    KL_start_nonlin = CEb_true_nonlin - belief_entropy

    KL_train_lin    = np.array(CEs_true_lin) - belief_entropy
    KL_train_nonlin = np.array(CEs_true_nonlin) - belief_entropy
    KL_train_lin    = KL_train_lin.tolist()
    KL_train_nonlin = KL_train_nonlin.tolist()

    # Build dataframe
    df = pd.DataFrame({
        "index": indices,
        "kl_linear": [KL_start_lin] + KL_train_lin,
        "kl_nonlinear": [KL_start_nonlin] + KL_train_nonlin,
        "tv_linear": [TVb_true_lin] + TVs_true_lin,
        "tv_nonlinear": [TVb_true_nonlin] + TVs_true_nonlin
    })

    # Save to CSV
    print(df)
    df.to_csv("divergence_results.csv", index=False)  
    print("\nSaved KL divergence results to divergence_results.csv")
