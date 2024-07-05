#!/bin/bash
# This script is used to run the generate_data.py file for creating training data.
# Creator: Anubhab Ghosh, Feb 2024.

# The python kernel version e.g. to run on python 3.8 version use: python3.8
PYTHON="python"

# The number of i.i.d. trajectories each of length T that constitute the training data
N=10

# Length of each such training data trajectory, default it is set to T=1000
T=10

# The name of the script for generating data with full path name
script_name="./bin/generate_data_rotnist.py"

# Output path to store the data
output_path="./data/rotnist"


${PYTHON} ${script_name} \
--num_samples $N \
--sequence_length $T \
--output_path ${output_path}
#--dataset_type ${dataset_type} \

