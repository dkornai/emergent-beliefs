"""Hyperparameters"""

OUTPUT_DIR = 'results/test'

# How much and what type of data to collect
SAVE_PARAM = False
REPLICATES = 3

# Model collection 
RNN_HIDDEN = 64
N_VALUE_MODELS = 1
N_Q_MODELS = 1

# Environment
N = 3
M = 5
SELF_TRANISTION = 0.1
GENERIC_REWARD  = -1.0
CLIFF_REWARD    = -10.0
TARGET_REWARD   = 10.0
GAMMA           = 0.98

# Data Gathering
EPISODES_PER_CHUNK  = 100
NUM_NEW_CHUNKS      = 100

# Training
ACTOR_STEPS  = 10
WORLD_STEPS  = 50
LAMBDA_ACTOR = 1.0
LAMBDA_VALUE = 1.0
LAMBDA_WORLD = 1.0
