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
    
    def update_state(self, prev_state: np.ndarray, action_i:int) -> int:
        """
        Update state via s_t ~ p(s_t | s_{t-1}, a_{t-1})
        """
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
    
    def update_belief(self, prev_belief_state: np.ndarray, obs_i:int, action: int) -> np.ndarray:
        """
        Belief update is via:
         p(x_t|b_t) propto p(o_t | s_t) \sum_{s_{t-1}} p(s_t | s_{t-1}, a_{t-1}) p(x_{t-1}|b_{t-1})
        """        
        belief_state = prev_belief_state @ self.tp_matrix[action]
        belief_state *= self.obs_matrix[:, obs_i]
        belief_state /= np.sum(belief_state)
        
        return belief_state
    
    def interact(self, prev_state, prev_belief, action_i:int) -> tuple[int, float, int, np.ndarray]:
        """
        Take a step in the environment with the given action.
        """
        new_state_i = self.update_state(prev_state, action_i)
        reward      = self.yield_reward(new_state_i)
        obs_i       = self.yield_observation(new_state_i)
        new_belief  = self.update_belief(prev_belief, obs_i, action_i)
        
        return new_state_i, obs_i, reward, new_belief
    
    def step(self, action_i:int):
        raise NotImplementedError("This method should be implemented in a subclass.")


class CliffWalk(PomdpEnv):
    """
    Partially Observable Cliff Walk Environment
    """
    def __init__(self, n=3, m=5, self_transition_prob=0.2, gamma=0.9, generic_reward=0.0, cliff_reward=-10.0, target_reward=10.0):
        
        self.n = n # Number of rows
        self.m = m # Number of columns
        self.state_dim = n * m # Total number of states
        
        
        self.self_transition_prob = self_transition_prob # Probability of staying in the same state
        self.action_space = [0, 1, 2, 3] # left, up, right, down
        self.action_dim = 4
        self.gamma = gamma # Discount factor

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

    def step(self, action_i:int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]:
        """
        Take a step in the environment with the given action.
        """
        if self.done:
            raise RuntimeError("Episode has reached termination state. Please reset env before taking a step.")
        
        # Update state, reward, observation, and belief state
        new_state_i, obs_i, reward, new_belief = self.interact(self.state, self.belief_state, action_i)
        
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


class InfiniteWalk(PomdpEnv):
    """
    Partially Observable Cliff Walk Environment
    """
    def __init__(self, self_transition_prob=0.2):
        self.n = 4 # Number of rows
        self.m = 4
        self.state_dim = 16 # Total number of states
        
        self.self_transition_prob = self_transition_prob # Probability of staying in the same state
        self.action_space = [0, 1, 2, 3] # left, up, right, down

        self.state = None
        self.belief_state = None
        self.done = False
        
        self.tp_matrix   = self.init_tp_matrix()
        self.reward_vec  = self.init_reward_vec()
        self.obs_matrix  = self.init_observation_matrix()

        self.obs_dim = self.obs_matrix.shape[1]  # Number of unique observations

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

                    if action == 2: # right
                        x_new = (x+1) % 4
                        y_new = y
                    elif action == 1: # up
                        y_new = (y+1) % 4
                        x_new = x
                    elif action == 3: # down
                        y_new = (y-1) % 4
                        x_new = x
                    elif action == 0: # left
                        x_new = (x-1) % 4
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
        reward_vec = np.full((self.state_dim), 0) # Default reward is generic reward
        
        return reward_vec

    def init_observation_matrix(self) -> np.ndarray:
        """
        Observation matrix for the environment. size is [|states|, |observations|].
        Start and end states are revealed as seperate observations [0, 1], otherwise only vertical position is revealed.
        """
        obs_matrix = np.zeros((self.state_dim, 4))
        for x in range(self.m):
            for y in range(self.n):
                current_state_index = x + y * self.m
                if x <= 1 and y <= 1:
                    obs_matrix[current_state_index, 0] = 1.0
                elif x <= 1 and y > 1:
                    obs_matrix[current_state_index, 1] = 1.0
                elif x > 1 and y <= 1:
                    obs_matrix[current_state_index, 2] = 1.0
                elif x > 1 and y > 1:
                    obs_matrix[current_state_index, 3] = 1.0
                    

        return obs_matrix

    def check_done(self):
        """
        Check if the current state is a terminal state.
        """
        # In this environment, there are no terminal states
        self.done = False    

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.done = False

        # Reset state to the start position in a random place
        quadrant = np.random.randint(0, 4)
        sub_quadrant = np.random.randint(0, 4)
        # Map quadrant to top-left corner of the 2x2 block
        quad_row = (quadrant // 2) * 2
        quad_col = (quadrant % 2) * 2

        # Map sub_quadrant to 2x2 position
        sub_row = sub_quadrant // 2
        sub_col = sub_quadrant % 2

        # Combine to get full grid coordinates
        row = quad_row + sub_row
        col = quad_col + sub_col

        # Compute flat index
        start_state_i = row * 4 + col
        
        self.state = onehot(self.state_dim, start_state_i)
        
        # Observation is the unique start position
        obs_i = np.random.choice(self.obs_dim, p=self.obs_matrix[start_state_i])
        observation = onehot(self.obs_dim, obs_i)

        # Reward generic at the start position
        reward = 0

        # Initialize the belief state in the quadrant of the start state
        self.belief_state = np.zeros((4, 4))
        # quadrant should have distributed belief
        if quadrant == 0:
            self.belief_state[0:2, 0:2] = 1.0 / 4
        elif quadrant == 1:
            self.belief_state[0:2, 2:4] = 1.0 / 4
        elif quadrant == 2:
            self.belief_state[2:4, 0:2] = 1.0 / 4
        elif quadrant == 3:
            self.belief_state[2:4, 2:4] = 1.0 / 4
        self.belief_state = self.belief_state.flatten()
        
        return self.state, observation, reward, self.belief_state, self.done

    def step(self, action_i:int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]:
        """
        Take a step in the environment with the given action.
        """
        if self.done:
            raise RuntimeError("Episode has reached termination state. Please reset env before taking a step.")
        
        # Update state, reward, observation, and belief state
        new_state_i, obs_i, reward, new_belief = self.interact(self.state, self.belief_state, action_i)
        
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