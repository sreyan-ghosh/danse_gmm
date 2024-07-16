#!/bin/bash
# This script is used to run the generate_data.py file for creating training data.
# Creator: Kasper Malm & Sreyan Ghosh, July 2024.

# The python kernel version e.g. to run on python 3.8 version use: python3.8
PYTHON="python"

# The name of the script for generating data with full path name
script_name="./src/rotnist_enc_dec.py"

# Output path to store the encoded data
output_path="./data/encoded_data"

# Output path to store the model
saved_model_path="./models/rotnist_models"

# train, encode, decode
mode="decode"

# vae, ae
model_type="ae"


${PYTHON} ${script_name} \
--output_path ${output_path} \
--saved_model_path ${saved_model_path} \
--mode ${mode} \
--model_type ${model_type} 