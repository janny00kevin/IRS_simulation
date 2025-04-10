# khatri_rao.py

import torch

def batch_khatri_rao(H_a, H_b):
    """
    Compute Khatri-Rao product in batch.

    Args:
    - H_a: Tensor of shape (data_size, n_ax, n_ay)
    - H_b: Tensor of shape (data_size, n_bx, n_by)
        Requires n_ay == n_by (column-wise compatibility).

    Returns:
    - Tensor of shape (data_size, n_ax * n_bx, n_ay), Khatri-Rao product.
    """
    data_size, n_ax, n_ay = H_a.shape
    _, n_bx, n_by = H_b.shape

    # Ensure column compatibility
    assert n_ay == n_by, "Khatri-Rao requires the last dimension of H_a and H_b to match (n_ay == n_by)."

    # Expand dimensions for broadcasting
    H_a_exp = H_a.unsqueeze(2).repeat(1, 1, n_bx, 1)  # Shape: (data_size, n_ax, n_bx, n_ay)
    H_b_exp = H_b.unsqueeze(1).repeat(1, n_ax, 1, 1)  # Shape: (data_size, n_ax, n_bx, n_ay)

    # Element-wise multiplication and reshape
    result = (H_a_exp * H_b_exp).reshape(data_size, n_ax * n_bx, n_ay)  # Shape: (data_size, n_ax * n_bx, n_ay)
    return result