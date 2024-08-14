#!/bin/bash
# This script is used to run the generate_data.py file for creating training data.
# Creator: Kasper Malm & Sreyan Ghosh, July 2024.

# The python kernel version e.g. to run on python 3.8 version use: python3.8
PYTHON="python"

# The number of training sequencess each of length T
N_train=500

# Length of each such training data sequence
T_train=60

# The number of testing sequencess each of length T
N_test=100

# Length of each such testing data sequence
T_test=60

# The name of the script for generating data with full path name
script_name="./bin/generate_data_rotnist.py"

# Output path to store the data
output_path="./data/rotnist"

rm -r ${output_path}/*

${PYTHON} ${script_name} \
--num_samples_tr $N_train \
--sequence_length_tr $T_train \
--num_samples_te $N_test \
--sequence_length_te $T_test \
--output_path ${output_path}
#--dataset_type ${dataset_type} \

