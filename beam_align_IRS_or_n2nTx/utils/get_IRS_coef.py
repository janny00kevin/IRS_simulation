import torch

def get_IRS_coef(IRS_coef_type,n_R,n_I,n_T,T):
        """
        Generate the IRS coefficient matrix based on the specified type.

        Args:
        - IRS_coef_type (str): Type of coefficient matrix ('identity', 'dft', 'hadamard').
        - n_I (int): Number of columns to extract.
        - T (int): Total size.
        - n_T (int): Divisor to calculate matrix size.

        Returns:
        - Tensor: The IRS coefficient matrix.

        Raises:
        - ValueError: If IRS_coef_type is not one of 'identity', 'dft', or 'hadamard'.
        - AssertionError: If the input conditions for the chosen type are not satisfied.
        """
        sqrt2 = 2**.5
        IRS_coef_type = IRS_coef_type.lower()
        N = T//n_T  # Size of the IRS coefficient matrix
        assert n_I <= N, 'n_I must be less than or equal to T//n_T for the IRS coefficient matrix.'

        if IRS_coef_type in ['identity', 'i']:
            # assert n_I == N, "The IRS coefficient matrix must be square if 'identity' is chosen."
            return torch.eye(N)[:n_I, :]
        
        elif IRS_coef_type in ['dft', 'd']:
            k = torch.arange(N).unsqueeze(1)  # Row indices (shape: N x 1)
            l = torch.arange(N).unsqueeze(0)  # Column indices (shape: 1 x N)
            # Compute the DFT matrix
            dft_matrix  = torch.exp(-2j * torch.pi * k * l / N)  # Shape: (N, N)
            return dft_matrix[:n_I, :] # Extract the first n_I rows
        
        elif IRS_coef_type in ['hadamard', 'h']:
            assert (N & (N - 1)) == 0, 'T//n_T must be a power of 2 to generate Hadamard matrix.'
            Hadamard = torch.tensor([[1]])
            # Recursive construction
            while Hadamard.size(0) < N:
                Hadamard = torch.cat([torch.cat([Hadamard, Hadamard], dim=1), torch.cat([Hadamard, -Hadamard], dim=1)], dim=0)
            return Hadamard[:n_I, :]
        
        elif IRS_coef_type in ['random', 'r']:
            return torch.view_as_complex(torch.normal(0, 1, size=(n_I, N, 2))/sqrt2)

        else:
            raise ValueError("IRS coefficient matrix should be 'identity'('i'), 'dft'('d') or 'hadamard'('h')")