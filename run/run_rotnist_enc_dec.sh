#!/bin/bash
# This script is used to run the rotnist_enc_dec.py file for creating encoded 
# and noised training and testing data for DANSE.
# 
# Creator: Kasper Malm & Sreyan Ghosh, July 2024.

# The python kernel version e.g. to run on python 3.8 version use: python3.8
PYTHON="python"

# The name of the script for generating data with full path name
script_name="./src/rotnist_enc_dec.py"

# Output path to store the model
saved_model_path="./models/rotnist_models"

# train, encode, decode, noise
mode="encode"

# train_danse, test_danse
danse_mode="test_danse"

# vae, ae
model_type="ae"

# n_obs and m_states
latent_dim="16"

if [ "$danse_mode" == "train_danse" ]; then

    output_path="./data/encoded_data/train_data"
    danse_input_path="./data/encoded_noise_data/train_data"
    
    # Run the training process once
    if [ "$mode" == "train" ]; then
        ${PYTHON} ${script_name} \
        --output_path ${output_path} \
        --danse_input_path ${danse_input_path} \
        --saved_model_path ${saved_model_path} \
        --mode ${mode} \
        --danse_mode ${danse_mode} \
        --model_type ${model_type} \
        --latent_dim ${latent_dim}
        mode="encode"  # Change mode to "encode" for subsequent runs
    fi

    # Run the encode process once
    if [ "$mode" == "encode" ]; then
        ${PYTHON} ${script_name} \
        --output_path ${output_path} \
        --danse_input_path ${danse_input_path} \
        --saved_model_path ${saved_model_path} \
        --mode ${mode} \
        --danse_mode ${danse_mode} \
        --model_type ${model_type} \
        --latent_dim ${latent_dim}
        mode="noise"  # Change mode to "noise" for the next run
    fi

    # Run the noise process 3 times
    if [ "$mode" == "noise" ]; then
        for smnr_dB in -5.0 0.0 5.0 10.0
        do
            ${PYTHON} ${script_name} \
            --output_path ${output_path} \
            --danse_input_path ${danse_input_path} \
            --saved_model_path ${saved_model_path} \
            --mode ${mode} \
            --danse_mode ${danse_mode} \
            --smnr_db ${smnr_dB} \
            --model_type ${model_type} \
            --latent_dim ${latent_dim}
        done

        # Delete files in the encoded_data directory
        rm -f ${output_path}/*
        echo "Deleted files in $output_path"
    fi
fi

if [ "$danse_mode" == "test_danse" ]; then

    output_path="./data/encoded_data/test_data"
    danse_input_path="./data/encoded_noise_data/test_data"

    # Run the encode process once
    if [ "$mode" == "encode" ]; then
        ${PYTHON} ${script_name} \
        --output_path ${output_path} \
        --danse_input_path ${danse_input_path} \
        --saved_model_path ${saved_model_path} \
        --mode ${mode} \
        --danse_mode ${danse_mode} \
        --model_type ${model_type} \
        --latent_dim ${latent_dim}
        mode="noise"  # Change mode to "noise" for the next run
    fi

    # Run the noise process 3 times
    if [ "$mode" == "noise" ]; then
        for smnr_dB in -5.0 0.0 5.0 10.0
        do
            ${PYTHON} ${script_name} \
            --output_path ${output_path} \
            --danse_input_path ${danse_input_path} \
            --saved_model_path ${saved_model_path} \
            --mode ${mode} \
            --danse_mode ${danse_mode} \
            --smnr_db ${smnr_dB} \
            --model_type ${model_type} \
            --latent_dim ${latent_dim}
        done

        # Delete files in the encoded_data directory
        rm -f ${output_path}/*
    fi
fi

if [ "$mode" == "decode" ]; then
    for smnr_dB in -5.0 0.0 5.0 10.0
    do
        ${PYTHON} ${script_name} \
        --output_path ${output_path} \
        --danse_input_path ${danse_input_path} \
        --saved_model_path ${saved_model_path} \
        --mode ${mode} \
        --danse_mode ${danse_mode} \
        --smnr_db ${smnr_dB} \
        --model_type ${model_type} \
        --latent_dim ${latent_dim}
    done
fi