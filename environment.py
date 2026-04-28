import numpy as np
import matplotlib.pyplot as plt

def onehot(size, index) -> np.ndarray:
    vec = np.zeros(size)
    vec[index] = 1.0
    return vec


class PomdpEnv():
    """
    Base class for POMDP environments with discrete state and observation spaces.
    Contains methods for calculating value and Q-value functions, updating state, yielding observations and rewards, and updating belief states.
    """
    def __init__(self):
        raise NotImplementedError("This is a base class. Please implement a dedicated method in a subclass.")
    
    def get_value_function(self, policy):
        """
        Calculate the value function under a given policy on the !latent! state space.
        """
        # Get the marginal transition probability matrix under the policy
        P_pi = np.einsum('sa, asn -> sn', policy, self.tp_matrix)
      
        # Get the reward vector under the policy using V = (I - P_pi)^-1 @ R
        I = np.eye(self.state_dim)
        V_pi = np.linalg.solve(I - self.gamma*P_pi, self.reward_vec)

        return np.round(V_pi, 2)
    
    def get_q_value_function(self, policy):
        """
        Calculate the Q-value function under a given policy on the !latent! state space.
        """
        # Get value function under the policy
        V_pi = self.get_value_function(policy)

        # Calculate Q-values via Q(s, a) = R(s) + sum_{s'} P(s'|s, a) V(s')
        Q_pi = np.zeros((self.state_dim, 4))
        Q_pi += self.reward_vec[:, None]
        Q_pi += self.gamma*np.einsum('asn, n -> sa', self.tp_matrix, V_pi)

        return np.round(Q_pi, 2)
    
    def update_state(self, prev_state: np.ndarray, action_oh:np.ndarray) -> int:
        """
        Update state via s_t ~ p(s_t | s_{t-1}, a_{t-1})
        """
        action_i = np.argmax(action_oh)
        state_probs = prev_state @ self.tp_matrix[action_i]
        state_index = np.random.choice(self.state_dim, p=state_probs)

        return state_index

    def yield_observation(self, state_i:int) -> int:
        """
        Observation index is given by o_t ~ p(o_t | s_t)
        """
        obs_probs   = self.obs_matrix[state_i]
        obs_index   = np.random.choice(self.obs_dim, p=obs_probs)
        
        return obs_index
    
    def yield_reward(self, state_i:int) -> float:
        """
        Reward is a scalar value associated with the state.
        """
        return self.reward_vec[state_i]
    
    def update_belief(self, prev_belief_state: np.ndarray, obs_i:int, action_oh: np.ndarray) -> np.ndarray:
        """
        Belief update is via:
         p(x_t|b_t) propto p(o_t | s_t) \sum_{s_{t-1}} p(s_t | s_{t-1}, a_{t-1}) p(x_{t-1}|b_{t-1})
        """        
        action_i = np.argmax(action_oh)
        belief_state = prev_belief_state @ self.tp_matrix[action_i]
        belief_state *= self.obs_matrix[:, obs_i]
        belief_state /= np.sum(belief_state)
        
        return belief_state
    
    def interact(self, prev_state, prev_belief, action_oh: np.ndarray) -> tuple[int, float, int, np.ndarray]:
        """
        Take a step in the environment with the given action.
        """
        new_state_i = self.update_state(prev_state, action_oh)
        reward      = self.yield_reward(new_state_i)
        obs_i       = self.yield_observation(new_state_i)
        new_belief  = self.update_belief(prev_belief, obs_i, action_oh)
        
        return new_state_i, obs_i, reward, new_belief
    
    def step(self, action_oh: np.ndarray):
        raise NotImplementedError("This method should be implemented in a subclass.")


class CliffWalk(PomdpEnv):
    """
    Partially Observable Cliff Walk Environment
    """
    def __init__(self, n=3, m=5, self_transition_prob=0.2, generic_reward=0.0, cliff_reward=-10.0, target_reward=10.0):
        
        self.n = n # Number of rows
        self.m = m # Number of columns
        self.state_dim = n * m # Total number of states
        
        
        self.self_transition_prob = self_transition_prob # Probability of staying in the same state
        self.action_space = [0, 1, 2, 3] # left, up, right, down
        self.action_dim = 4
        self.gamma = 0.98 # Discount factor

        self.generic_reward = generic_reward
        self.cliff_reward   = cliff_reward
        self.target_reward  = target_reward

        self.state = None
        self.belief_state = None
        self.done = False
        
        self.tp_matrix   = self.init_tp_matrix()
        self.reward_vec  = self.init_reward_vec()
        self.obs_matrix  = self.init_observation_matrix()

        self.obs_dim = self.obs_matrix.shape[1]  # Number of unique observations

        self.actions_discrete = True
        self.obs_discrete = True

        self.reset()

    def init_tp_matrix(self) -> np.ndarray:
        """
        Transition Probability Matrix (TPM) for the environment.
        the TPM is a 3D tensor of size [|actions|, |states|, |states|]
        """
        tpm = np.zeros((len(self.action_space), self.state_dim, self.state_dim))
        for x in range(self.m):
            for y in range(self.n):
                current_state_index = x + y * self.m
                for action in self.action_space:
                    # Terminal states have 0 outgoing transitions
                    if y == 0 and x > 0:
                        continue
                    
                    # Non-terminal states
                    else:
                        if action == 2: # right
                            x_new = min(x+1, self.m-1)
                            y_new = y
                        elif action == 1: # up
                            y_new = min(y+1, self.n-1)
                            x_new = x
                        elif action == 3: # down
                            y_new = max(y-1, 0)
                            x_new = x
                        elif action == 0: # left
                            x_new = max(x-1, 0)
                            y_new = y

                        target_state_index = x_new + y_new * self.m
                        # add probability of moving to the target state
                        tpm[action, current_state_index, target_state_index]    += (1 - self.self_transition_prob)
                        # add probability of staying in the same state
                        tpm[action, current_state_index, current_state_index]   += self.self_transition_prob

        return tpm

    def init_reward_vec(self) -> np.ndarray:
        """
        Reward vector for the environment. size is [|states|].
        """
        reward_vec = np.full((self.state_dim), self.generic_reward) # Default reward is generic reward
        for x in range(self.m):
            for y in range(self.n):
                if (x == self.m - 1 and y == 0):
                    reward_vec[x + y * self.m] = self.target_reward # Goal state
                elif y == 0 and x > 0 and x < self.m - 1:
                    reward_vec[x + y * self.m] = self.cliff_reward  # Cliff states
        
        return reward_vec

    def init_observation_matrix(self) -> np.ndarray:
        """
        Observation matrix for the environment. size is [|states|, |observations|].
        Start and end states are revealed as seperate observations [0, 1], otherwise only vertical position is revealed.
        """
        obs_matrix = np.zeros((self.state_dim, self.n+2))
        for x in range(self.m):
            for y in range(self.n):
                current_state_index = x + y * self.m
                
                # Start state has unique observation
                if x == 0 and y == 0:
                    obs_matrix[current_state_index, 0] = 1.0
                
                # Goal state has unique observation
                elif y == 0 and x == self.m - 1:
                    obs_matrix[current_state_index, 1] = 1.0
                
                # Otherwise, only vertical position is revealed
                else:
                    obs_matrix[current_state_index, y + 2] = 1.0
        
        ## Fully observable case (for debugging purposes)
        # obs_matrix = np.zeros((self.state_dim, self.state_dim))
        # for x in range(self.m):
        #     for y in range(self.n):
        #         current_state_index = x + y * self.m
        #         obs_matrix[current_state_index, current_state_index] = 1.0

        return obs_matrix
    
    def get_optimal_policy(self, epsilon=0.0):
        """
        Get the optimal policy for the environment, optionally with epsilon-greedy exploration.
        """
        pi = np.zeros((self.state_dim, len(self.action_space)))
        for x in range(self.m):
            for y in range(self.n):
                current_state_index = x + y * self.m
                
                # Start state, move up
                if x == 0 and y == 0:
                    pi[current_state_index, 1] = 1.0
                
                # Any non-cliff state, not at the right edge, move right
                elif y > 0 and x < self.m - 1:
                    pi[current_state_index, 2] = 1.0
                
                # in terminal states (cliff and goal) all actions are equally likely
                elif y == 0 and x > 0:
                    pi[current_state_index, :] = 0.25
                
                # right edge, move down
                else:
                    pi[current_state_index, 3] = 1.0

        # Add epsilon-greedy exploration
        if epsilon > 0.0:
            for state_index in range(self.state_dim):
                pi[state_index] = (1 - epsilon) * pi[state_index] + (epsilon / 4)
        
        return pi

    def check_done(self):
        """
        Check if the current state is a terminal state.
        """
        # Terminal states are the cliff and the goal state
        if np.argmax(self.state) in list(range(1, self.m)):
            self.done = True

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.done = False

        # Reset state to the start position (0, 0)
        self.state = onehot(self.state_dim, 0)
        
        # Observation is the unique start position
        observation = onehot(self.obs_dim, 0)

        # Reward generic at the start position
        reward = self.generic_reward

        # Initialize the fully resolved belief state at the start position (due to the revealing observation)
        self.belief_state = onehot(self.state_dim, 0)
        
        return self.state, observation, reward, self.belief_state, self.done

    def step(self, action_oh:np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]:
        """
        Take a step in the environment with the given action.
        """
        if self.done:
            raise RuntimeError("Episode has reached termination state. Please reset env before taking a step.")
        
        # Update state, reward, observation, and belief state
        new_state_i, obs_i, reward, new_belief = self.interact(self.state, self.belief_state, action_oh)
        
        self.state  = onehot(self.state_dim, new_state_i)
        self.belief_state = new_belief
        observation = onehot(self.obs_dim, obs_i)

        # Check if the episode is done
        self.check_done()

        return self.state, observation, reward, self.belief_state, self.done

    def render(self):
        """
        Render the environement"
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # render the position of the agent
        axs[0].imshow(self.state.reshape((self.n, self.m)), cmap='gray', vmin=0, vmax=1)
        axs[0].set_title("Agent Position")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].invert_yaxis()
        # render the belief state
        axs[1].imshow(self.belief_state.reshape((self.n, self.m)), cmap='gray', vmin=0, vmax=1)
        axs[1].set_title("Belief State")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].invert_yaxis()
        plt.show()

# DEFAULT ENVIRONMENT PARAMETERS (for easy access in other files)
cw_default_params_dict = {
    "n": 3,
    "m": 5,
    "self_transition_prob": 0.1,
    "generic_reward": -1.0,
    "cliff_reward": -10.0,
    "target_reward": 10.0
}