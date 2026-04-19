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



"""
Environment wrapper for the dm_control Reacher task.

Wraps the dm_control suite reacher into an interface compatible with the
emergent-beliefs training loop. Key differences from the discrete PomdpEnv:

  - Observations are continuous float vectors (not one-hot).
  - Actions are continuous float vectors in [-1, 1] (not discrete indices).
  - There is no exact Bayesian belief; the "true state" (qpos, qvel) is
    recorded instead and used as the decoding target.
  - Episodes are fixed-length (no terminal absorbing states).

dm_control reacher specs (easy / hard):
  Observations (flattened):
    - position : cos/sin of 2 joint angles → 4 floats
    - to_target : 2D vector from finger to target → 2 floats
    - velocity  : 2 joint angular velocities → 2 floats
    Total obs_dim = 8   (when all three keys are included; 6 without velocity
                         — see `include_velocity` flag)

  Actions:
    - 2 joint torques, each in [-1, 1]
    Total action_dim = 2

  True state (decoding target, always includes velocity):
    - qpos : 2 joint angles                       → 2 floats
    - qvel : 2 joint angular velocities            → 2 floats
    - (optional) target (x, y) position            → 2 floats
    Total state_dim = 4  (joints only) or 6 (joints + target pos)
    Which components to include is controlled by `state_components`.

  Return signature (mirrors CliffWalk):
    reset() → (state, observation, reward, state, done)
    step(a) → (state, observation, reward, state, done)
    The true state occupies both the first and fourth (belief_state) slots.

  Reward:
    - Continuous scalar in [0, 1] based on distance to target.
"""

try:
    from dm_control import suite
except ImportError:
    raise ImportError(
        "dm_control is required for ReacherEnv. "
        "Install via: pip install dm_control"
    )


class ReacherEnv:
    """
    Wraps dm_control reacher into the interface expected by the
    emergent-beliefs training loop.

    The public API mirrors the discrete PomdpEnv / CliffWalk:
        reset()  → (state, observation, reward, state, done)
        step(a)  → (state, observation, reward, state, done)

    The return signature preserves the 5-tuple layout of CliffWalk's
    (state, observation, reward, belief_state, done).  Since there is no
    exact Bayesian belief for this environment, the true physics state
    appears in both the first and fourth positions.
    """

    def __init__(
        self,
        task: str = "easy",
        max_steps: int = 50,
        include_velocity: bool = True,
        state_components: str = "joints",
    ):
        """
        Parameters
        ----------
        task : str
            'easy' (large target) or 'hard' (small target).
        max_steps : int
            Maximum number of environment steps per episode.  dm_control's
            default time_limit for reacher is 1000 steps (1 s at 0.02 dt
            × 1000 = 20 s with action_repeat=1, but the suite default is
            time_limit=1 s with 0.02 control_timestep → 50 steps).  Set
            this to override or cap episode length.
        include_velocity : bool
            If True the observation vector includes joint velocities
            (obs_dim = 8).  If False, only position + to_target are used
            (obs_dim = 6), making the task partially observable.
        state_components : str
            Which physics quantities form the "true state" used as the
            decoding target.
              'joints'      → qpos[:2] (joint angles) + qvel[:2]  → dim 4
              'joints+target' → above + target (x,y)               → dim 6
        """
        # ---- Build dm_control environment ----
        self._env = suite.load(
            domain_name="reacher",
            task_name=task,
        )

        self.actions_discrete = False
        self.obs_discrete = False
        self.action_space = [0, 1] # dummy action space (not used since actions are continuous, but need to get action_dim from it)
        self.action_dim = 2

        self.task = task
        self.max_steps = max_steps
        self.include_velocity = include_velocity
        self.state_components = state_components

        # ---- Inspect specs once ----
        obs_spec = self._env.observation_spec()
        act_spec = self._env.action_spec()

        # Observation dimension
        # Keys returned by reacher: 'position' (4), 'to_target' (2), 'velocity' (2)
        self._obs_keys = ["position", "to_target"]
        if self.include_velocity:
            self._obs_keys.append("velocity")

        self.obs_dim = sum(obs_spec[k].shape[0] for k in self._obs_keys)

        # Action dimension and bounds
        self.action_dim = act_spec.shape[0]          # 2
        self.action_low = act_spec.minimum.copy()     # [-1, -1]
        self.action_high = act_spec.maximum.copy()    # [ 1,  1]

        # True-state dimension (used as decoding target)
        if self.state_components == "joints":
            # joint angles (2) + joint velocities (2)
            self.state_dim = 4
        elif self.state_components == "joints+target":
            # joint angles (2) + joint velocities (2) + target xy (2)
            self.state_dim = 6
        else:
            raise ValueError(
                f"Unknown state_components='{state_components}'. "
                "Use 'joints' or 'joints+target'."
            )

        # ---- Runtime bookkeeping ----
        self._step_count = 0
        self._time_step = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, time_step) -> np.ndarray:
        """
        Extract and concatenate the chosen observation keys into a single
        flat float32 vector.
        """
        parts = [time_step.observation[k] for k in self._obs_keys]
        return np.concatenate(parts).astype(np.float32)

    def _extract_state(self) -> np.ndarray:
        """
        Extract the true physics state used as the decoding target.

        For 'joints':
            [shoulder_angle, elbow_angle, shoulder_vel, elbow_vel]

        For 'joints+target':
            above + [target_x, target_y]
        """
        physics = self._env.physics

        # Joint angles and velocities (first 2 entries of qpos/qvel;
        # the remaining qpos entries are the target body position).
        qpos_joints = physics.data.qpos[:2].copy()
        qvel_joints = physics.data.qvel[:2].copy()

        parts = [qpos_joints, qvel_joints]

        if self.state_components == "joints+target":
            # Target body position is stored in the remaining qpos entries
            # (indices 2 and 3 in the default reacher model).
            target_xy = physics.data.qpos[2:4].copy()
            parts.append(target_xy)

        return np.concatenate(parts).astype(np.float32)

    def _get_reward(self, time_step) -> float:
        """Extract scalar reward from a dm_control TimeStep."""
        reward = time_step.reward
        if reward is None:
            return 0.0
        return float(reward)

    def _is_done(self, time_step) -> bool:
        """
        Episode ends when dm_control says it's the last step OR
        we have exceeded max_steps.
        """
        return time_step.last() or self._step_count >= self.max_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        state : np.ndarray, shape (state_dim,)
            True physics state.
        observation : np.ndarray, shape (obs_dim,)
            Agent-visible observation vector.
        reward : float
            Initial reward (0.0 at reset).
        state : np.ndarray, shape (state_dim,)
            True physics state (occupies the belief_state slot).
        done : bool
            Always False after reset.
        """
        self._time_step = self._env.reset()
        self._step_count = 0

        state = self._extract_state()
        observation = self._flatten_obs(self._time_step)
        reward = 0.0
        done = False

        return state, observation, reward, state, done

    def step(self, action: np.ndarray):
        """
        Take one environment step.

        Parameters
        ----------
        action : np.ndarray, shape (action_dim,)
            Continuous action vector.  Will be clipped to [action_low,
            action_high].

        Returns
        -------
        state : np.ndarray, shape (state_dim,)
            True physics state.
        observation : np.ndarray, shape (obs_dim,)
            Agent-visible observation vector.
        reward : float
            Scalar reward for this step.
        state : np.ndarray, shape (state_dim,)
            True physics state (occupies the belief_state slot).
        done : bool
            Whether the episode has ended.
        """
        # Clip to valid range (actor may overshoot due to numerical issues
        # even after tanh).
        action = np.clip(action, self.action_low, self.action_high)

        self._time_step = self._env.step(action)
        self._step_count += 1

        state = self._extract_state()
        observation = self._flatten_obs(self._time_step)
        reward = self._get_reward(self._time_step) * 10  # Scale reward to [0, 10] for better learning dynamics (tune as needed)
        done = self._is_done(self._time_step)

        return state, observation, reward, state, done

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    def sample_random_action(self) -> np.ndarray:
        """Return a uniformly random action within the valid bounds."""
        return np.random.uniform(
            self.action_low, self.action_high
        ).astype(np.float32)

    @property
    def physics(self):
        """Direct access to the underlying MuJoCo physics, for debugging."""
        return self._env.physics

    def __repr__(self):
        return (
            f"ReacherEnv(task={self.task!r}, max_steps={self.max_steps}, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"state_dim={self.state_dim}, "
            f"include_velocity={self.include_velocity})"
        )