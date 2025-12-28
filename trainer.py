import os

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from environment import CliffWalk
from nn_models import ModelCollection
from train_chunk import train_with_chunks

# Import parameters from config file
import config


CURRENT_DIR = os.getcwd()
os.mkdir(f"{config.OUTPUT_DIR}")
os.chdir(f"{config.OUTPUT_DIR}")

def experiment(rep):
    os.mkdir(f"{rep}")
    os.chdir(f"{rep}")
    
    print("\n\n>>>>>>>>>>>>  Starting Process <<<<<<<<<<<<<<\n")
    # 1) Initialise the POMDP environment
    env = CliffWalk(
        n                       = config.N, 
        m                       = config.M, 
        self_transition_prob    = config.SELF_TRANISTION,
        gamma                   = config.GAMMA,
        generic_reward          = config.GENERIC_REWARD,
        cliff_reward            = config.CLIFF_REWARD,
        target_reward           = config.TARGET_REWARD
        )  
    

    # 2) Initialise the model and optimizers
    models = ModelCollection(
        latent_dim              = config.RNN_HIDDEN,
        n_actions               = env.action_dim ,
        n_obs                   = env.obs_dim,
        n_value_models          = config.N_VALUE_MODELS,
        n_q_models              = config.N_Q_MODELS
        )
    models.to(DEVICE)

    optimizers = models.init_optimizers()
    

    # 3) Run the training loop 
    models = train_with_chunks(
        env                     = env,
        models                  = models,
        optimizers              = optimizers,
        num_new_chunks          = config.NUM_NEW_CHUNKS,
        ep_per_chunk            = config.EPISODES_PER_CHUNK,
        gamma                   = config.GAMMA,
        actor_steps             = config.ACTOR_STEPS,
        world_steps             = config.WORLD_STEPS,
        lambda_actor            = config.LAMBDA_ACTOR,
        lambda_value            = config.LAMBDA_VALUE,
        lambda_world            = config.LAMBDA_WORLD,
        device                  = DEVICE,
        save_checkp             = config.SAVE_PARAM
        )

    print("Training complete!")
    os.chdir("..")

# Run the 
for i in range(config.REPLICATES):
    experiment(i)

os.chdir(CURRENT_DIR)