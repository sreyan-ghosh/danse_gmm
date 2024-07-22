import numpy as np
import glob
import torch
from torch import nn
import math
from torch.utils.data import DataLoader, Dataset
import sys
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.functional import jacobian
from parse import parse
from timeit import default_timer as timer
import json
import tikzplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils_rotnist import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss, \
    mse_loss_dB, load_saved_dataset, save_dataset, nmse_loss_std, mse_loss_dB_std, NDArrayEncoder, partial_corrupt, get_dataloaders, \
    Series_Dataset
#from parameters import get_parameters, A_fn, h_fn, f_lorenz_danse, f_lorenz_danse_ukf, delta_t, J_test
from config.parameters_opt import get_parameters, A_fn, h_fn, f_lorenz_danse, f_lorenz_danse_ukf, delta_t, J_test
#from src.k_net import KalmanNetNN
from src.danse_rotnist import DANSE, push_model

splits = load_saved_dataset("data/encoded_noise_data/splits_m_32_n_32_rotnist_T_20_N_500_smnr_20.0dB.pkl")
tr_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]
datafile = "data/encoded_noise_data/sequence_m_32_n_32_rotnist_T_20_N_500_smnr_20.0dB.pkl"
Z_XY = load_saved_dataset(filename=datafile)
Z_XY_dataset = Series_Dataset(Z_XY_dict=Z_XY)

test_data = [Z_XY_dataset[idx] for idx in test_indices]

# Extract the components, inner format might be numpy and cause errors
Z = torch.cat([torch.tensor(sample["targets"]) for sample in test_data], dim=0)
Y = torch.cat([torch.tensor(sample["inputs"]) for sample in test_data], dim=0)
Cw_i = torch.cat([torch.tensor(sample["Cw"]) for sample in test_data], dim=0)

print(type(Z[0][0]))