# DANSE: Data-driven Non-linear State Estimation of Model-free Process in Unsupervised Learning Setup

This is the repository for implementing a nonlinear state estimation of a model-free process with Linear measurements 

Pre-print: *Anubhab Ghosh, Antoine Honoré, and Saikat Chatterjee. "DANSE: Data-driven Non-linear State Estimation of Model-free Process in Unsupervised Learning Setup." arXiv preprint arXiv:2306.03897 (2023)* [https://arxiv.org/abs/2306.03897](https://arxiv.org/abs/2306.03897)

**Accepted in IEEE Transactions on Signal Processing (IEEE-TSP) (March 2024)**

## Authors
Anubhab Ghosh (anubhabg@kth.se), Antoine Honoré (honore@kth.se)

## Dependencies
You can create a virtual environment with the required packages and activate it using [`conda`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) :
```bash
conda env create -f environment.yml
conda activate danse_env
```

## Datasets used 

The experiments were carried out using synthetic data generated with linear and non-linear SSMs:

- Linear state space models (Linear SSMs)
- Non-linear state space models (Non-linear SSMs): In our case, we used chaotic attractors:
    - Lorenz attractor 
    - Chen attractor
    - Lorenz-96 attractor

Details about these models and their underlying dynamics can be found in `./bin/ssm_models.py`. 

## Reference models (implemented in PyTorch + Numpy)

- Kalman filter (KF)
- Extended Kalman filter (EKF)
- Unscented Kalman filter (UKF)
- Unsupervised KalmanNet
    - The model implementation was adapted from [https://github.com/KalmanNet/Unsupervised_EUSIPCO_22](https://github.com/KalmanNet/Unsupervised_EUSIPCO_22)
    - Hyper-parameters for individual experiments were also taken from another repository of the same authors: [https://github.com/KalmanNet/KalmanNet_TSP](https://github.com/KalmanNet/KalmanNet_TSP)
- Deep Markov model (DMM)
    - The original implementation was done in Theano (no longer maintained ML framework for Python): [https://github.com/clinicalml/dmm](https://github.com/clinicalml/dmm)
    - We used a Pytorch implementation instead: [https://github.com/yjlolo/pytorch-deep-markov-model/blob/master/](https://github.com/yjlolo/pytorch-deep-markov-model/blob/master/)
    - Hyper-parameters were taken from the original implementation in Theano

## GPU Support

The training-based methods: DANSE, DMM and KalmanNet, were run on a single NVIDIA-Tesla P100 GPU with 16 GB of memory. 

## Code organization
This would be the required organization of files and folders for reproducing results. If certain folders are not present, they should be created at that level.

````
- main_danse_opt.py (main function for calling training 'DANSE' model)
- main_kalmannet.py (main function for calling reference training 'KalmanNet' model)
[more mains]

- data/ (contains stored datasets in .pkl files)
| - synthetic_data/ (contains datasets related to SSM models in .pkl files)

- src/ (contains model-related files)
| - danse.py (for training the unsupervised version of DANSE)
| - danse_supervised.py (for training the supervised version of DANSE, refer to section 2.E of the paper)
| - kf.py (for running the Kalman filter (KF) at test-time for inference)
| - ekf.py (for running the extended Kalman filter (EKF) at test-time for inference)
| - ukf_aliter.py (for running the unscented Kalman filter (UKF) at test-time for inference)
| - ukf_aliter_one_step.py (for running the unscented Kalman filter (UKF) at test-time for inference related to one-step ahead of forecasting!)
| - k_net.py (for training the unsupervised KalmanNet model)
| - dmm_causal.py (for training the deep Markov model with structured-approximation (DMM-ST-L))
| - rnn.py (Class definition of the RNN model for DANSE)

- log/ (contains training and evaluation logs, losses in `.json`, `.log` files)
- models/ (contains saved model checkpoints as `.pt` files)
- figs/ (contains resulting model figures)
- utils/ (contains helping functions)
- tests/ (contains files and functions for evaluation at test time)
- config/ (contains the parameter file)
| - parameters_opt.py (Python file containing relevant parameters for different architectures)

- bin/ (contains data generation files)
| - ssm_models.py (contains class for state space models)
| - generate_data.py (contains code for generating training datasets)

- run/ (folder containing the shell scripts to run the `main` scripts at one go for either different smnr_dB / sigma_e2_dB / N)
| - run_main_danse.sh 
| - run_main_knet.sh
````

## Brief outline of DANSE training

1. Generate data by calling `bin/generate_data.py`. This can be done in a simple manner by editing and calling the shell script `run_generate_data.sh`. Data gets stored at `data/synthetic_data/`. For e.g. to generate trajectory data with 1000 samples with each trajectory of length 100, from a Lorenz Attractor model (m=3, n=3), with $\sigma_{e}^{2}= -10$ dB, and $\text{SMNR}$ = $0$ dB, the syntax should be 
````
python3 ./bin/generate_data.py --n_states 3 --n_obs 3 --num_samples 1000 --sequence_length 100 --sigma_e2_dB -10 --smnr 0 --dataset_type LorenzSSM --output_path ./data/synthetic_data/
````
2. Edit the hyper-parameters for the DANSE architecture in `./config/parameters_opt.py`.

3. Run the training for DANSE by calling `main_danse_opt.py`.  E.g. to run a DANSE model employing a GRU architecture as the RNN, using the Lorenz attractor dataset as described above, the syntax should be 
```
python3 main_danse_opt.py \
--mode train \
--rnn_model_type gru \
--dataset_type LorenzSSM \
--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_N_1000_T_100_sigmae2_-10.0dB_smnr_0.0dB.pkl \
--splits ./data/synthetic_data/splits_m_3_n_3_LorenzSSM_data_N_1000_T_100_sigmae2_-10.0dB_smnr_0.0dB.pkl
```
For the `datafile` and `splits` arguments:
`N` denotes the number of sample trajectories, `T` denotes the length of each sample trajectory. 

4. To reproduce experiments, for multiple SMNRs, run the shell script `./run/run_main_danse.sh`.

5. Similarly, other `main_...` files and corresponding scripts can be used to reproduce experiments for kalmannet and dmm.


### Grid-search (for architectural choice of DANSE)

Can be run by calling the script `main_danse_gs.py` with grid-search parameters to be edited in the script directly. 

## Evaluation

Once files are created, the evaluation can be done by calling scripts in `/tests/`. Paths to model files and log files should be edited in the script directly. 
More information in this [readme](https://github.com/anubhabghosh/danse_jrnl/blob/main/tests/readme.md).
