"""Hyperparameters"""

OUTPUT_DIR = 'results/res_reach'

# How much and what type of data to collect
SAVE_PARAM = True
REPLICATES = 1

# Environment type
ENV_TYPE = 'reacher'
GAMMA = 0.98

# Model collection 
RNN_HIDDEN = 128
N_VALUE_MODELS = 1
N_Q_MODELS = 1

# Data Gathering
EPISODES_PER_CHUNK  = 500
NUM_NEW_CHUNKS      = 101

# Training
ACTOR_STEPS  = 5
WORLD_STEPS  = 50
LAMBDA_ACTOR = 1.0
LAMBDA_VALUE = 0.0
LAMBDA_WORLD = 0.5
