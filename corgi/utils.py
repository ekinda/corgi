import torch
from torch.nn.functional import poisson_nll_loss
from torch.cuda import memory_allocated, memory_reserved, max_memory_allocated
import numpy as np
import logging

def poisson_nll_masked(
    y_pred: torch.Tensor,                  # shape (b, 22, 8192)
    y_true: torch.Tensor,
    mask: torch.Tensor,                    # shape (b, 22, 1)
    epsilon: float = 1e-6
):
    poisson_loss = poisson_nll_loss(y_pred, y_true, log_input=False, eps=epsilon, reduction="none")
    channel_losses = (poisson_loss * mask).mean(dim=2)
    return channel_losses       # (b, 22)
    
def poisson_multinomial_masked(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,                    # shape (b, 22, 1)
    total_weight: float = 0.2,
    epsilon: float = 1e-6,
    channel_weights: torch.Tensor = None   # 1-d tensor
):
    """
    Code modified from github/calico/baskerville and github/gagneurlab/scooby

    Calculates the Poisson-Multinomial loss, while masking some features.

    This loss function combines a Poisson loss term for the total count and a multinomial loss term for the 
    distribution across sequence positions.

    Args:
        y_pred (torch.Tensor): Predicted values (batch_size, features, seq_len).
        y_true (torch.Tensor): True values (batch_size, features, seq_len).
        mask (torch.Tensor): 0-1 mask that indicates channels that contribute to loss computation. (batch_size, features, 1)
        total_weight (float, optional): Weight of the Poisson total term. Defaults to 0.2.
        epsilon (float, optional): Small value added to avoid log(0). Defaults to 1e-6.
        channel_weights: (torch.Tensor, shape (22,). To rescale the loss of different channels, before taking the mean.

    Returns:
        torch.Tensor: The mean Poisson-Multinomial loss.
    """

    mask = mask.squeeze(-1)

    seq_len = y_true.shape[-1]

    # add epsilon to protect against tiny values
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    # sum across lengths
    s_true = y_true.sum(dim=-1, keepdim=False)  # (B, F)
    s_pred = y_pred.sum(dim=-1, keepdim=False)  # (B, F)

    # normalize to sum to one
    p_pred = y_pred / s_pred.unsqueeze(-1)      # (B, F, L)

    # total count poisson loss, masked
    poisson_term = poisson_nll_loss(
        (s_pred * mask) + (epsilon * (1-mask)),
        (s_true * mask) + (epsilon * (1-mask)),
        log_input=False, eps=0, reduction="none")  # B x F
    poisson_term /= seq_len

    # multinomial loss
    log_p_pred = torch.log(p_pred)  # (B x F X L)
    multinomial_term = -torch.sum(y_true * log_p_pred, dim=-1)  # (B, F)
    multinomial_term /= seq_len

    # Combine terms (per-track)
    loss = (multinomial_term + total_weight * poisson_term) * mask  # Masked

    if channel_weights is not None:
        loss = loss * channel_weights.unsqueeze(0)

    n_available = mask.sum()
    if n_available == 0:
        return torch.tensor(0.0, device=y_true.device)
    else:
        # Normalize by total available tracks across all samples
        return loss.sum() / n_available

def poisson_multinomial_masked_v2(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,                    # shape (b, f, 1)
    total_weight: float = 0.2,
    epsilon: float = 1e-6
):
    mask = mask.squeeze(-1)
    seq_len = y_true.shape[-1]
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon
    s_true = y_true.sum(dim=-1, keepdim=False)  # (B, F)
    s_pred = y_pred.sum(dim=-1, keepdim=False)  # (B, F)
    p_pred = y_pred / s_pred.unsqueeze(-1)      # (B, F, L)
    poisson_term = poisson_nll_loss(
        (s_pred * mask) + (epsilon * (1-mask)),
        (s_true * mask) + (epsilon * (1-mask)),
        log_input=False, eps=0, reduction="none")  # (B, F)
    poisson_term /= seq_len

    log_p_pred = torch.log(p_pred)  # (B, F, L)
    multinomial_term = -torch.sum(y_true * log_p_pred, dim=-1)  # (B, F)
    multinomial_term /= seq_len

    loss = (multinomial_term + total_weight * poisson_term) * mask  # Masked
    return loss         # (B, F)

def load_experiment_mask(mask_path):
    """
    Loads the available experiment mask from a numpy array
    and converts it to a dictionary of form {tissue_id:mask, where mask is a numpy array}
    """
    data = np.load(mask_path)

    tissue_mask_dict = {}
    for t_id, mask in enumerate(data):
        tissue_mask_dict[t_id] = mask
    return tissue_mask_dict

def log_gpu_usage(stage):
    if torch.cuda.is_available():
        logging.info(
            f"[{stage}] GPU memory allocated: {memory_allocated() / 1e6:.2f} MB, "
            f"reserved: {memory_reserved() / 1e6:.2f} MB, "
            f"max allocated: {max_memory_allocated() / 1e6:.2f} MB"
        )

def base_to_one_hot(base_byte):
    """
    Converts a DNA base (e.g., b'A') into a one-hot np.array([1,0,0,0]) etc.
    Returns an np.array of shape (4,).
    """
    base = base_byte.upper()
    if base == b'A':
        return np.array([1, 0, 0, 0], dtype=np.float32)
    elif base == b'C':
        return np.array([0, 1, 0, 0], dtype=np.float32)
    elif base == b'G':
        return np.array([0, 0, 1, 0], dtype=np.float32)
    elif base == b'T':
        return np.array([0, 0, 0, 1], dtype=np.float32)
    else:
        return np.array([0, 0, 0, 0], dtype=np.float32)

def one_hot_to_base(one_hot_arr):
    """
    Converts a one-hot np.array of shape (4,) to a base character 'A', 'C', 'G', or 'T'.
    """
    if not (one_hot_arr.shape == (4,)):
        raise ValueError(f"one_hot_to_base expects shape (4,), got {one_hot_arr.shape}")
    idx = np.argmax(one_hot_arr)
    if one_hot_arr[idx] == 0:
        # No '1' in the array
        return 'N'
    return 'ACGT'[idx]