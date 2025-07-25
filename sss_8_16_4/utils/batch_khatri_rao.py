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
    _, n_bx, _ = H_b.shape

    # Ensure column compatibility
    assert n_ay == H_b.shape[2], "Khatri-Rao requires the last dimension of H_a and H_b to match (n_ay == n_by)."
    assert data_size == H_b.shape[0], "Batch Khatri-Rao requires the first dimension of H_a and H_b to match (data_size)."
    # print("H_a",H_a)
    # print("H_b",H_b)

    # Expand dimensions for broadcasting
    H_a_exp = H_a.unsqueeze(2).repeat(1, 1, n_bx, 1)  # Shape: (data_size, n_ax, n_bx, n_ay)
    H_b_exp = H_b.unsqueeze(1).repeat(1, n_ax, 1, 1)  # Shape: (data_size, n_ax, n_bx, n_ay)
    # print("H_a_exp",H_a_exp)
    # print("H_b_exp",H_b_exp)

    # Element-wise multiplication and reshape
    result = (H_a_exp * H_b_exp).reshape(data_size, n_ax * n_bx, n_ay)  # Shape: (data_size, n_ax * n_bx, n_ay)
    return result

def batch_khatri_rao_chunked(H_a, H_b, chunk_size=32):
    """
    分块计算 batch Khatri–Rao，以降低显存峰值。

    Args:
      - H_a: Tensor of shape (B, I, K)
      - H_b: Tensor of shape (B, J, K)
      - chunk_size: 每次处理的 K 维度块大小

    Returns:
      - Tensor of shape (B, I*J, K)
    """
    B, I, K = H_a.shape
    _, J, _ = H_b.shape

    out_chunks = []
    # 按 K 维度分块
    for start in range(0, K, chunk_size):
        end = min(K, start + chunk_size)
        a_chunk = H_a[:, :, start:end]     # (B, I, C)
        b_chunk = H_b[:, :, start:end]     # (B, J, C)
        # 广播乘法，不用 repeat，直接 unsqueeze
        # 结果形状 (B, I, J, C)
        mul = a_chunk.unsqueeze(2) * b_chunk.unsqueeze(1)
        # reshape 到 (B, I*J, C)
        out_chunks.append(mul.reshape(B, I*J, end - start))

        # 立即释放中间变量（可选）
        del mul, a_chunk, b_chunk

    # 按列拼回 (B, I*J, K)
    return torch.cat(out_chunks, dim=2)