import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
from tqdm import tqdm
# import h5py
from utils.batch_khatri_rao import batch_khatri_rao
from utils.IRS_rayleigh_channel import importData
from utils.complex_utils import turnReal, turnCplx, vec
import os


print("cuda:0", torch.cuda.is_available()) 