import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from episodes import EpisodeCollection

class BeliefDecoder(nn.Module):
    """
    Base class for belief decoders.
    """
    def __init__(self):
        super().__init__()


class LinBeliefDecoder(BeliefDecoder):
    """
    Linear belief decoder using a single linear layer.
    """
    def __init__(self, input_dim, belief_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, belief_dim)

        self.input_dim = input_dim
        self.belief_dim = belief_dim

    def forward(self, x):
        x = self.fc(x)  # [B, T, belief_dim]
        x = nn.functional.softmax(x, dim=-1)  # Apply softmax to get probabilities
        
        return x

class NonLinBeliefDecoder(BeliefDecoder):
    """
    Non-linear belief decoder using a feedforward neural network.
    """
    def __init__(self, input_dim, hidden_dim, belief_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, belief_dim)

        self.input_dim = input_dim  
        self.belief_dim = belief_dim

    def forward(self, x):
        x = self.fc1(x)        # works on [B, T, D]
        x = torch.relu(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=-1)  # [B, T, belief_dim]
        
        return x



def sequence_cross_entropy(p_target, q_pred, mask=None, eps=1e-8):
    """
    Compute cross-entropy between two categorical distributions over time.

    Args:
        p_target: [B, T, C] true distribution (e.g., one-hot or soft label)
        q_pred:   [B, T, C] predicted distribution (must be softmaxed already)
        mask:     [B, T] optional mask for valid time steps (1 = valid, 0 = ignore)
        eps:      small constant to avoid log(0)

    Returns:
        Scalar cross-entropy loss
    """
    # Ensure numerical stability
    log_q = torch.log(q_pred + eps)
    
    # Element-wise cross entropy: -P * log(Q)
    cross_entropy = -torch.sum(p_target * log_q, dim=-1)  # [B, T]

    if mask is not None:
        cross_entropy = cross_entropy * mask  # mask out invalid steps
        return cross_entropy.sum() / mask.sum()
    else:
        return cross_entropy.mean()
    
def sequence_tv_distance(p_target, q_pred, mask=None):
    """
    Compute total variation distance between two categorical distributions over time.

    Args:
        p_target: [B, T, C] true distribution (e.g., one-hot or soft label)
        q_pred:   [B, T, C] predicted distribution (must be softmaxed already)
        mask:     [B, T] optional mask for valid time steps (1 = valid, 0 = ignore)

    Returns:
        Scalar total variation distance
    """
    tv_distance = 0.5 * torch.sum(torch.abs(p_target - q_pred), dim=-1)  # [B, T]

    if mask is not None:
        tv_distance = tv_distance * mask  # mask out invalid steps
        return tv_distance.sum() / mask.sum()
    else:
        return tv_distance.mean()
    

def train_belief_decoder(
        belief_model:   BeliefDecoder,
        episodes:       EpisodeCollection, 
        input_states:   torch.Tensor, 
        belief_index    = [0, None], 
        num_epochs      = 100, 
        lr              = 1e-3, 
        batch_size      = 1000
        ):
    """
    Train the belief decoder using the RNN model's outputs as input.
    """
    # Validate input types
    assert isinstance(belief_model, BeliefDecoder), "belief_model must be an instance of BeliefDecoder"
    assert isinstance(episodes, EpisodeCollection), "episodes must be an instance of EpisodeCollection"
    assert isinstance(input_states, torch.Tensor), "input_states must be a torch.Tensor"
    
    # Set up input and target tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
    
    input_states   = input_states.to(device).contiguous()   # [B, T, hidden_dim]
    true_beliefs    = episodes.batch_beliefs[:, :, belief_index[0]:belief_index[1]].to(device).contiguous()   # [B, T, belief_dim]
    traj_mask       = episodes.batch_mask_traj.to(device).contiguous()   # [B, T]
    
    # Validate dimensions
    assert input_states.shape[0] == true_beliefs.shape[0], "Batch size of hidden states and true beliefs must match."
    assert belief_model.input_dim == input_states.shape[-1], f"Belief model input dimension {belief_model.input_dim} does not match hidden states dimension {input_states.shape[-1]}."
    assert belief_model.belief_dim == true_beliefs.shape[-1], f"Belief model output dimension {belief_model.belief_dim} does not match true beliefs dimension {true_beliefs.shape[-1]}."
    assert input_states.shape[1] == true_beliefs.shape[1], "Time steps of hidden states and true beliefs must match."
    
    # Set up the belief extractor model
    belief_model = belief_model.to(device)
    optimizer = torch.optim.Adam(belief_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    belief_model.train()

    # Training loop
    for epoch in range(num_epochs):
        if epoch % 50 == 0: ### THIS MIGHT BITE ME LATER WITH SMALLER DATASETS
            if input_states.shape[0] <= 5000:
                slice_index_start = 0
                slice_index_stop = input_states.shape[0]
            else:
                slice_index_start = np.random.randint(0, input_states.shape[0] - 5000)  # Randomly select a slice index
                slice_index_stop = slice_index_start + 5000
            input_states_batch  = input_states[slice_index_start:slice_index_stop].contiguous()  # Get a batch of input states
            true_beliefs_batch  = true_beliefs[slice_index_start:slice_index_stop].contiguous()  # Get a batch of true beliefs
            traj_mask_batch     = traj_mask[slice_index_start:slice_index_stop].contiguous()  # Get a batch of trajectory mask

        # Get beliefs from the belief decoder
        optimizer.zero_grad()
        pred_beliefs = belief_model(input_states_batch)  # shape: [B, T, belief_dim]
        
        # Calculate loss
        loss = sequence_cross_entropy(true_beliefs_batch, pred_beliefs, traj_mask_batch)
        tv = sequence_tv_distance(true_beliefs_batch, pred_beliefs, traj_mask_batch)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch+1}, CE Loss: {loss.item():.4f}, TV Distance: {tv.item():.4f}", end='\r')

    # Move the models back to CPU
    belief_model = belief_model.to('cpu')

    return loss.item(), tv.item()


def evaluate_belief(value_model, belief_model, episode):
    value_model.eval()  # switch to eval mode
    belief_model.eval()  # switch to eval mode

    with torch.no_grad():
        history = torch.tensor(episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]

        hidden = value_model(history)  # shape: [1, T]
        predicted_beliefs = belief_model(hidden)
        
        return predicted_beliefs.numpy()
    


def plot_decoded_belief_over_true(beliefs, test_episode, cliff_dim = (3, 4), belief_index = [0, None]):
    """
    Plot the decoded beliefs over non-nuiseance states (e.g., cliff states) and compare them with the true beliefs.
    """
    true_beliefs = test_episode.belief_states
    true_beliefs = np.round(true_beliefs, 2)
    # cut the true beliefs to the specified dimensions along the observation dimension
    true_beliefs = true_beliefs[:, belief_index[0]:belief_index[1]]  # [T, belief_dim]

    if cliff_dim is not None:
        assert beliefs.shape[1] == cliff_dim[0] * cliff_dim[1], "Belief dimensions do not match the cliff dimensions."

    for t in range(len(true_beliefs)):
        fig, axs = plt.subplots(1, 3, figsize=(8, 4))
        
        # True beliefs
        axs[0].imshow(true_beliefs[t].reshape(cliff_dim), cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        axs[0].set_title(f"True Beliefs at t={t}")
        axs[0].axis('off')
        axs[0].invert_yaxis()  # Invert y-axis to match the grid orientation
        
        # Predicted beliefs        
        axs[1].imshow(beliefs[t].reshape(cliff_dim), cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        axs[1].set_title(f"Decoded Beliefs at t={t}")
        axs[1].axis('off')
        axs[1].invert_yaxis()
        
        # Compute and display the difference
        diff_data = np.abs(true_beliefs[t] - beliefs[t])
        diff_data = diff_data.reshape(cliff_dim)
        axs[2].imshow(diff_data, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        axs[2].set_title(f"Difference at t={t}")
        axs[2].axis('off')
        axs[2].invert_yaxis()

        # Annotate the cells in the third subplot
        for i in range(cliff_dim[0]):
            for j in range(cliff_dim[1]):
                axs[2].text(j, i, f"{diff_data[i, j]:.2f}", ha='center', va='center', color='white')

        plt.tight_layout()
        plt.show()


def plot_decoded_belief_over_nuisance(beliefs, test_episode, belief_index = [0, None]):
    """
    Plot the decoded beliefs over nuisance states (HMM output) and compare them with the true beliefs.
    """
    true_beliefs = test_episode.belief_states
    true_beliefs = np.round(true_beliefs, 2)
    # cut the true beliefs to the specified dimensions along the observation dimension
    true_beliefs = true_beliefs[:, belief_index[0]:belief_index[1]]  # [T, belief_dim]

    fig, axs = plt.subplots(1, 3, figsize=(8, 8))
    
    # True beliefs
    axs[0].imshow(true_beliefs, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    axs[0].set_title("True Beliefs")
    axs[0].axis('off')
    
    # Predicted beliefs
    axs[1].imshow(beliefs, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    axs[1].set_title("Decoded Beliefs")
    axs[1].axis('off')
    
    # Compute and display the difference
    diff_data = np.abs(true_beliefs - beliefs)
    diff_data = diff_data.reshape(true_beliefs.shape)
    axs[2].imshow(diff_data, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axs[2].set_title("Difference")
    axs[2].axis('off')
    # Annotate the cells in the third subplot
    for i in range(diff_data.shape[0]):
        for j in range(diff_data.shape[1]):
            axs[2].text(j, i, f"{diff_data[i, j]:.2f}", ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()


def decode_training(episodes, belief_decoder, indices, value_RNN=None, num_epochs=1001, lr=1e-3, batch_size=5000):

    if value_RNN is not None:
        histories   = episodes.batch_histories
        mask_traj   = episodes.batch_mask_traj
        with torch.no_grad():
            input_to_belief_decoder = value_RNN(histories)
    else:
        input_to_belief_decoder = episodes.batch_histories
    
    ce_loss, tv_loss = train_belief_decoder(belief_decoder, episodes, input_to_belief_decoder, num_epochs=num_epochs, lr=lr, batch_size=batch_size, belief_index=indices)
    print("CE Loss:", np.round(ce_loss, 2), "TV Loss:", np.round(tv_loss, 2), "                                 ")

    return belief_decoder, ce_loss, tv_loss
    
def decode_visualisation(test_episode, belief_decoder, indices, env_size, value_RNN=None):    
    
    if value_RNN is not None:
        pred_belief = evaluate_belief(value_RNN, belief_decoder, test_episode)[0]
    else:
        pred_belief = belief_decoder(torch.tensor(test_episode.history, dtype=torch.float32)).detach().numpy()
    
    print("Belief Decoder:")
    if indices[1] is None:
        plot_decoded_belief_over_true(pred_belief, test_episode, env_size, belief_index=indices)
    else:
        plot_decoded_belief_over_nuisance(pred_belief, test_episode, belief_index=indices)

def estimate_entropy(belief_states, base=np.e):
    eps = 1e-8  # prevent log(0)
    belief_states = np.clip(belief_states, eps, 1.0)
    log_fn = np.log if base == np.e else lambda x: np.log2(x) if base == 2 else lambda x: np.log(x) / np.log(base)
    
    entropies = -np.sum(belief_states * log_fn(belief_states), axis=1)
    return np.round(np.mean(entropies),2)