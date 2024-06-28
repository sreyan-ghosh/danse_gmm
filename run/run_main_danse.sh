#!/bin/bash
# This script is used to run the main_danse_opt.py or the main_danse_supervised_opt.py file for running DANSE.
# Creator: Anubhab Ghosh, Feb 2024.

# The python kernel version e.g. to run on python 3.8 version use: python3.8
PYTHON="python"

# The number of i.i.d. trajectories each of length T that constitute the training data
N=500

# Length of each such training data trajectory, default it is set to T=1000
T=1000

# Number of hidden states in the process, usually for Lorenz (a.k.a. Lorenz-63), Chen 
# attractors, the number of hidden states is equal to 3, while for Lorenz-96, this value must be changed to
# n_states= 20 (currently hardcoded in this manner) but can be in general n_states >= 4
n_states=3

# Number of observations in the measurement system
n_obs=3

# dataset_type defines the type of dynamical system, the general terminology, e.g. for the Lorenz 63 system, 
# the type is LorenzSSM, similarly for Chen attractor we have ChenSSM.
# For the Lorenz-96 model, we have Lorenz96SSM.
# For underdetermined scenario with random matrix subsampled measurements: LorenzSSMrn${n_obs}, ChenSSMrn${n_obs},
# Lorenz96SSMrn${n_obs} with deterministic matrix subsampled measurements: LorenzSSMn${n_obs}, ChenSSMn${n_obs}, 
# Lorenz96SSMn${n_obs}.
# For the linear system, we have LinearSSM (can handle both full-rank, deterministic downsampled case).
dataset_type="LorenzSSM"

# The name of the script for generating data with full path name. 
script_name="./main_danse_opt.py" # (or ./main_danse_supervised_opt.py)

# Output path to store the data
output_path="./data/synthetic_data/"

# Set the process noise level (in dB)
sigma_e2_dB=-10.0

# RNN model type (e.g. GRU / LSTM)
rnn_model_type="gru"

for smnr_dB in -10.0 0.0 10.0 20.0 30.0
do
	${PYTHON} ${script_name} \
	--mode test \
	--rnn_model_type ${rnn_model_type} \
	--model_file_saved "models\LorenzSSM_danse_opt_gru_m_3_n_3_T_1000_N_500_sigmae2_-10.0dB_smnr_$(echo $smnr_dB)dB\danse_gru_ckpt_epoch_671_best.pt"\
	--dataset_type ${dataset_type} \
	--datafile ${output_path}/trajectories_m_${n_states}_n_${n_obs}_${dataset_type}_data_T_${T}_N_${N}_sigmae2_${sigma_e2_dB}dB_smnr_$(echo $smnr_dB)dB.pkl \
	--splits ${output_path}/splits_m_${n_states}_n_${n_obs}_${dataset_type}_data_T_${T}_N_${N}_sigmae2_${sigma_e2_dB}dB_smnr_$(echo $smnr_dB)dB.pkl
done
