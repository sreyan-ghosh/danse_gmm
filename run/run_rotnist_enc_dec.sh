#!/bin/bash
# This script is used to run the generate_data.py file for creating training data.
# Creator: Kasper Malm & Sreyan Ghosh, July 2024.

# The python kernel version e.g. to run on python 3.8 version use: python3.8
PYTHON="python"

# The name of the script for generating data with full path name
script_name="./src/rotnist_enc_dec.py"

# Output path to store the encoded data: encoded_data, encoded_noise_data
output_path="./data/encoded_data"

danse_input_path="./data/encoded_noise_data"

# Output path to store the model
saved_model_path="./models/rotnist_models"

# train, encode, decode
mode="noise"

# vae, ae
model_type="ae"

for smnr_dB in 10.0 20.0 30.0 
do
    ${PYTHON} ${script_name} \
    --output_path ${output_path} \
    --danse_input_path ${danse_input_path} \
    --saved_model_path ${saved_model_path} \
    --mode ${mode} \
    --smnr_db ${smnr_dB} \
    --model_type ${model_type} 
done

