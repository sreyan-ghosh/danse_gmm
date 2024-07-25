import pickle as pkl
import numpy as np

filename = "data/encoded_noise_data/test_data/test_sequence_m_32_n_32_rotnist_T_20_N_100_smnr_0.0dB.pkl"
with open(filename, 'rb') as handle:
        zxy = pkl.load(handle)
fpaths = zxy["img_paths"]

fpaths = np.asarray(fpaths)
print(fpaths[0])