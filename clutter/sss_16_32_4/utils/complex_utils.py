import torch

def turn_real(data):
    """
    turn a cplx vector into real, eg. [1+2j 3+4j] -> [1 2 3 4]
    """
    cplx = 2
    size = data.shape
    return torch.view_as_real(data).reshape(size[0],size[1]*cplx)

def turn_cplx(data):
    """
    turn a real vector into cplx, eg. [1 2 3 4] -> [1+2j 3+4j]
    """
    cplx = 2
    size = data.shape
    return torch.view_as_complex(data.reshape(size[0],size[1]//cplx,cplx).contiguous())

def vec(data):
    """
    vectorize a batch of matrix
    """
    size = data.shape
    return data.permute(0, 2, 1).reshape(size[0], size[1]*size[2])