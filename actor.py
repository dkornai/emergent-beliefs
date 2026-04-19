import torch
import numpy as np

from episodes import Episode
from nn_models import BeliefRNN, ActorReadout

class ActorPolicyWrapper:
    """
    Wraps BeliefRNN + ActorMLP into a callable policy object suitable
    for collect_episodes(...) to use.
    Performs online GRU filtering, sampling stochastic actions.
    """

    def __init__(self, belief_rnn: BeliefRNN, actor: ActorReadout, device="cpu"):
        self.belief_rnn = belief_rnn
        self.actor = actor
        self.device = device
        self.hidden = None  # RNN hidden state for online filtering

    def reset(self):
        """Call at the beginning of every episode."""
        self.hidden = None

    @torch.no_grad()
    def __call__(self, observation, prev_action):
        """
        observation : numpy array (one-hot)
        prev_action : numpy array (one-hot)
        
        Returns:
            action_index : int
            log_prob     : float (optional; can be returned/stored)
        """
        # 1. Prepare input vector for RNN: concat(o_t, a_{t-1})
        inp = torch.tensor(
            np.concatenate([observation, prev_action]),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0).unsqueeze(0)  # shape [1, 1, H]

        # 2. Forward Belief RNN one step
        z_t, self.hidden = self.belief_rnn.rnn(inp, self.hidden)
        z_t = z_t.squeeze(0).squeeze(0)  # → [latent_dim]

        # 3. Sample action from Actor MLP
        action = self.actor.sample_action(z_t)  # numpy array, either one-hot or raw values

        return action


def collect_episodes_actor(env, actor_policy: ActorPolicyWrapper, num_episodes: int):
    """
    Collect episodes using the actor policy wrapper.
    This version replaces the tabular policy pathway.
    Returns a list of Episode objects.
    """
    episodes = []

    for _ in range(num_episodes):
        ep = Episode()

        state, observation, reward, belief, done = env.reset()

        # Must reset RNN hidden state for a fresh episode
        actor_policy.reset()

        prev_action = np.zeros(len(env.action_space))

        while not done:
            # Add step data
            ep.add_step(state, observation, reward, prev_action, belief)

            # Query actor policy wrapper
            action = actor_policy(observation, prev_action)
            prev_action = action  # Update prev_action for next step

            # Step env
            state, observation, reward, belief, done = env.step(action)

        # Add final step
        ep.add_step(state, observation, reward, prev_action, belief)
        ep.finish_episode()

        episodes.append(ep)

    return episodes
