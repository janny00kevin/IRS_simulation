def batch_kronecker(A, B):
    """
    Compute Kronecker Product in batch or directly for 2D matrices.

    Args:
    - A: Tensor of shape (data_size, n_a, m_a) or (n_a, m_a)
    - B: Tensor of shape (data_size, n_b, m_b) or (n_b, m_b)

    Returns:
    - Tensor of shape (data_size, n_a * n_b, m_a * m_b) for batch input,
      or (n_a * n_b, m_a * m_b) for 2D matrices.
    """
    if A.ndim == 2 and B.ndim == 2:  # If both are 2D matrices
        n_a, m_a = A.shape
        n_b, m_b = B.shape
        # print("A shape:", A.shape)
        # print("B shape:", B.shape)


        # Expand dimensions for broadcasting
        A_exp = A.unsqueeze(1).unsqueeze(3)  # Shape: (n_a, 1, m_a, 1)
        B_exp = B.unsqueeze(0).unsqueeze(2)  # Shape: (1, n_b, 1, m_b)
        # print("A_exp shape:", A_exp.shape)
        # print("B_exp shape:", B_exp.shape)


        # Perform element-wise multiplication
        result = (A_exp * B_exp).reshape(n_a * n_b, m_a * m_b)  # Shape: (n_a * n_b, m_a * m_b)
        return result

    elif A.ndim == 3 and B.ndim == 3:  # If both are batched matrices
        data_size, n_a, m_a = A.shape
        _, n_b, m_b = B.shape

        # Ensure the batch dimensions match
        assert A.shape[0] == B.shape[0], "Batch Kronecker Product requires the first dimension of A and B to match (data_size)."

        # Expand dimensions for broadcasting
        A_exp = A.unsqueeze(3).unsqueeze(2)  # Shape: (data_size, n_a, 1, m_a, 1)
        B_exp = B.unsqueeze(1).unsqueeze(3)  # Shape: (data_size, 1, n_b, 1, m_b)

        # Perform element-wise multiplication
        result = (A_exp * B_exp).reshape(data_size, n_a * n_b, m_a * m_b)  # Shape: (data_size, n_a * n_b, m_a * m_b)
        return result

    else:
        raise ValueError("Inputs must either both be 2D matrices or both be 3D batched tensors.")
