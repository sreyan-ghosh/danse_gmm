# DANSE: Data-driven Non-linear State Estimation of Model-free Process in Unsupervised Learning Setup

This is the repository for implementing a nonlinear state estimation of a model-free process with Linear measurements 

Pre-print: [https://arxiv.org/abs/2306.03897](https://arxiv.org/abs/2306.03897)

**Accepted in IEEE Transactions on Signal Processing (IEEE-TSP) (March 2024)**

## Authors
Anubhab Ghosh (anubhabg@kth.se), Antoine Honoré (honore@kth.se), Saikat Chatterjee (sach@kth.se)

## Dependencies

It is recommended to build an environment either in [`pip`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [`conda`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) and install the following packages (I used `conda` as personal preference):
- PyTorch (1.6.0)
- Python (>= 3.7.0) with standard packages as part of an Anaconda installation such as Numpy, Scipy, Matplotlib etc. The settings for the code were:
    - Numpy (1.20.3)
    - Matplotlib (3.4.3)
    - Scipy (1.7.3)
    - Scikit-learn (1.0.1)

- Filterpy (1.4.5) (for implementation of Unscented Kalman Filter (UKF)): [https://filterpy.readthedocs.io/en/latest/](https://filterpy.readthedocs.io/en/latest/)
- Jupyter notebook (>= 6.4.6) (for result analysis)
- Tikzplotlib (for figures) [https://github.com/nschloe/tikzplotlib](https://github.com/nschloe/tikzplotlib)

## Reference models (implemented in PyTorch + Numpy)

- Kalman filter (KF)
- Extended Kalman filter (EKF)
- Unscented Kalman filter (UKF)
- Unsupervised KalmanNet
    - The code was adopted from the repository of the authors: [https://github.com/KalmanNet/Unsupervised_EUSIPCO_22](https://github.com/KalmanNet/Unsupervised_EUSIPCO_22)
    - Experimental details taken also from the repository of the supervised KalmanNet: [https://github.com/KalmanNet/KalmanNet_TSP](https://github.com/KalmanNet/KalmanNet_TSP)
- Deep Markov model (DMM)
    - The code was adopted from the repository: [https://github.com/yjlolo/pytorch-deep-markov-model/blob/master/](https://github.com/yjlolo/pytorch-deep-markov-model/blob/master/)
    - Experimental details needed to be taken from the official repo in Theano (Theano is no longer maintained): [https://github.com/clinicalml/dmm](https://github.com/clinicalml/dmm)
## Code organization

This would be required organization of files and folders for reproducing results. If certain folders are not present, they should be created at that level.

````
- data/ (contains stored datasets in .pkl files)
- src/ (contains model related files)
| - danse.py (for training the unsupervised version of DANSE)
| - danse_supervised.py (for training the supervised version of DANSE, refer to section 2.E of the paper)
| - kf.py (for running the Kalman filter (KF) at test-time for inference)
| - ekf.py (for running the extended Kalman filter (EKF) at test-time for inference)
| - ukf_aliter.py (for running the unscented Kalman filter (UKF) at test-time for inference)
| - ukf_aliter_one_step.py (for running the unscented Kalman filter (UKF) at test-time for inference related to one-step ahead of forecasting!)
| - k_net.py (for training the unsupervised KalmanNet model)
| - dmm_causal.py (for training the deep Markov model with structured-approximation (DMM-ST-L))

|···
- log/ (contains training and evaluation logs, losses in `.json`, `.log` files)
- models/ (contains saved model checkpoints saved as `.pt` files)
- figs/ (contains resulting model figures)
- utils/ (contains helping functions for \src\, etc.)
- tests/ (contains files and functions for evaluation at test time)
- parameters_opt.py (Pythnon file containing relevant parameters for different architectures)
- main_danse_opt.py (main function for calling training 'DANSE' model)
- main_kalmannet.py (main function for calling reference training 'KalmanNet' model)
- ssm_models.py (contains code for implementing state space models)
- generate_data.py (contains code for generating training datasets)
````

## Brief outline of DANSE training

1. Generate data by calling `generate_data.py`. This can be done in a simple manner by editing and calling the shell script `run_generate_data.sh`. Data gets stored at `data/synthetic_data/`. For e.g. to generate trajectory data with 500 samples with each trajetcoyr of length 1000, from a Lorenz Attractor model (m=3, n=3), with $\sigma_{e}^{2}= -20$ dB, and $\text{SMNR}$ = $0$ dB, the syntax should be 
````
[python interpreter e.g. python3.8] generate_data.py --n_states 3 --n_obs 3 --num_samples 500 --sequence_length 1000 --sigma_e2_dB -20 --smnr -20 --dataset_type LorenzSSM --output_path [dataset location e.g. ./data/synthetic_data/] \
````

The state space models are coded in `ssm_moddels.py`, and mainly for nonlinear SSMs, the LorenzSSM class is required. The parameter `alpha` decides whether to simulate a Lorenz attractor (`alpha=0`) or a Chen attractor (`alpha=1`)

2. Edit the parameters as per user choice to set architecture for DANSE in `parameters_opt.py`.

3. Run the training for DANSE by calling `main_danse.py`. This can be done in a simple manner by editing and calling the shell script 
`run_main_danse.sh`. Ensure that directories `/log/` and `/models/` have been created. E.g. to run a DANSE model employing a GRU architecture as the RNN, using the Lorenz attractor dataset as described above, the syntax should be 
```
[python interpreter e.g. python3.8] main_danse.py \
--mode train \
--rnn_model_type gru \
--dataset_type LorenzSSM \
--datafile [full path to dataset, e.g. ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_N_500_T_1000_sigmae2_-20.0dB_smnr_10.0dB.pkl] \
--splits ./data/synthetic_data/splits_m_3_n_3_LorenzSSM_data_N_500_T_1000_sigmae2_-20.0dB_smnr_10.0dB.pkl
```
`N` denotes the number of sample trajectories, `T` denotes the length of each sample trajectory. In the `.sh` file, you may find also the variable `n_obs` that denotes the dimension of the observation vector (denoted by $n$ in the paper). The number of states (`n_states`) for the `LorenzSSM` models should be always kept at 3. Assume also, for most cases `n_obs = 3`.

4. Run the training for the unsupervised KalmanNet by calling `main_kalmannet.py`. Also posible in similar manner as `run_main_knet_gpu1.sh`. Parameters have to be edited in `parameters.py`.

### Grid-search (for architectural choice of DANSE)

Can be run by calling the script `main_danse_gs.py` with grid-search parameters to be edited in the script directly. Relevant shell script that can be edited and used: `run_main_danse_gs_gpu1.sh`.

## Evaluation

Once files are created, the evaluation can be done by calling scripts in `/tests/`. Paths to model files and log files should be edited in the script directly. 

````
[python interpreter e.g. python3.8] tests/test_ukf_ekf_danse.py
````

1. Linear model ($2 \times 2$ model) comparison: `/tests/test_kf_linear_2x2.py`
2. Lorenz model comparsion: `/tests/test_ukf_ekf_danse.py`.
3. Chen model comparison: `/tests/test_ukf_ekf_danse_Chen.py`
