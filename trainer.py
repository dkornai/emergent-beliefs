import os
import argparse
from types import SimpleNamespace

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from environment import CliffWalk, cw_default_params_dict, ReacherEnv
from nn_models import ModelCollection
from train_chunk import train_with_chunks
from train_belief import belief_test


def load_config(filepath):
    """
    Load a config file that contains variable assignments (e.g., ENV_TYPE = 'cliffwalk').
    Ignores comments and blank lines. Returns a SimpleNamespace with the variables as attributes.
    """
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
                continue
            if '=' in line:
                line = line.split('#')[0].strip()
                key, value = line.split('=', 1)
                config[key.strip()] = eval(value.strip())
                
    return SimpleNamespace(**config)




if __name__ == "__main__":
    # Read config file path from command line arguments
    parser = argparse.ArgumentParser(description="Train belief state models with specified config.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file (e.g., configs/no_aux_config.py)")
    args = parser.parse_args()

    # Load config variables
    config = load_config(args.config)

    # Print loaded config for verification
    print("Loaded configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")

    # Create output directory for results
    CURRENT_DIR = os.getcwd()
    os.mkdir(f"{config.OUTPUT_DIR}")
    os.chdir(f"{config.OUTPUT_DIR}")

    # Define the experimental protocol for each replicate
    def experiment(rep):
        os.mkdir(f"{rep}")
        os.chdir(f"{rep}")
        
        print("\n\n>>>  Starting Process <<<\n")
        # 1) Initialise the POMDP environment
        if config.ENV_TYPE == 'cliffwalk':
            env = CliffWalk(**cw_default_params_dict)
        elif config.ENV_TYPE == 'reacher':
            env = ReacherEnv()
        else:
            raise ValueError(f"Unknown ENV_TYPE: {config.ENV_TYPE}")
        

        # 2) Initialise the model and optimizers
        models = ModelCollection(
            latent_dim              = config.RNN_HIDDEN,
            dim_actions             = env.action_dim ,
            actions_discrete        = env.actions_discrete,
            dim_obs                 = env.obs_dim,
            obs_discrete            = env.obs_discrete,
            n_value_models          = config.N_VALUE_MODELS,
            n_q_models              = config.N_Q_MODELS
            )
        models.to(DEVICE)

        optimizers = models.init_optimizers()
        

        # 3) Run the training loop 
        print("\n> Training started!")
        train_with_chunks(
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
        print("> Training complete!")
        
        # 4) Analyse the belief states
        print("\n> Belief analysis started!")
        belief_test(config)
        print("> Belief analysis complete!")

        os.chdir("..")

    # Run the replicates
    for i in range(config.REPLICATES):
        experiment(i)

    os.chdir(CURRENT_DIR)