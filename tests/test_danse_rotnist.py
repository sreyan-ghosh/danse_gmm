#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################

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

# Import for decoding:
from src.rotnist_enc_dec import AE, decode_data

from utils.plot_functions import *
from utils.utils_rotnist import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss, \
    mse_loss_dB, load_saved_dataset, save_dataset, nmse_loss_std, mse_loss_dB_std, NDArrayEncoder, partial_corrupt, \
    get_dataloaders, Series_Dataset, push_to_device
#from parameters import get_parameters, A_fn, h_fn, f_lorenz_danse, f_lorenz_danse_ukf, delta_t, J_test
from config.parameters_opt import get_parameters, A_fn, h_fn, f_lorenz_danse, f_lorenz_danse_ukf, delta_t, J_test, get_H_DANSE
#from src.k_net import KalmanNetNN
from src.danse_rotnist import DANSE, push_model
import argparse
from matplotlib.backends.backend_pdf import PdfPages

# Change:
def recreate_latent_values(mean, covariance, device):
    """
    Recreates latent dimension values from the given mean and covariance.

    Args:
        mean (torch.Tensor): The mean tensor of shape (N, latent_dim).
        covariance (torch.Tensor): The covariance tensor of shape (N, latent_dim, latent_dim).
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: The sampled latent dimension values of shape (N, latent_dim).
    """
    # Can directly use the posterior mean as the estimate, z_hat is the posterior mean
    N, T, latent_dim = mean.shape

    # Ensure the covariance matrix is positive semi-definite
    covariance = covariance + 1e-6 * torch.eye(latent_dim).to(device)

    # Sample from multivariate normal distribution
    z_samples = []
    for i in range(N):
        mvn = torch.distributions.MultivariateNormal(mean[i], covariance[i])
        z_sample = mvn.sample()
        z_samples.append(z_sample)

    z_samples = torch.stack(z_samples)
    return z_samples

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
def test_danse_rotnist(danse_model, saved_model_file, Y, Cw, device=None):

    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        Cw_test_batch = Variable(Cw, requires_grad=False).type(torch.FloatTensor).to(device)
        Z_estimated_pred, Pk_estimated_pred, Z_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch, Cw_test_batch)
    
    return Z_estimated_pred, Pk_estimated_pred, Z_estimated_filtered, Pk_estimated_filtered

def test_rotnist(device=None, model_file_saved=None, test_data_file=None, test_logfile=None, evaluation_mode='Full', p=0.5, bias=30):

    dataset_type, rnn_type, m, n, T, _, smnr_dB = parse("{}_danse_opt_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_smnr_{:f}dB", model_file_saved.split('/')[-2])

    delta = delta_t # If decimate is True, then set this delta to 1e-5 and run it for long time
    delta_d = 0.02
    J = 5
    J_test = 5
    #smnr_dB = 20
    decimate=False
    use_Taylor = False 

    orig_stdout = sys.stdout
    f_tmp = open(test_logfile, 'a')
    sys.stdout = f_tmp

    if not os.path.isfile(test_data_file):
        
        print('Dataset is not present, creating at {}'.format(test_data_file))
        # My own data generation scheme
        m, n, T_test, N_test, smnr_dB_test = parse("test_sequence_m_{:d}_n_{:d}_rotnist_T_{:d}_N_{:d}_smnr_{:f}dB.pkl", test_data_file.split('/')[-1])
        
        data_par_dir = "data/encoded_noise_data/train_data"
        splits_filepath = os.path.join(data_par_dir, f"splits_m_{m}_n_{n}_rotnist_T_{T_test}_N_{N_test}_smnr_{smnr_dB_test}dB.pkl")
        splits = load_saved_dataset(splits_filepath)
        tr_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]
        datafile = os.path.join(data_par_dir, f"sequence_m_{m}_n_{n}_rotnist_T_{T_test}_N_{N_test}_smnr_{smnr_dB_test}dB.pkl")
        Z_XY = load_saved_dataset(filename=datafile)
        Z_XY_dataset = Series_Dataset(Z_XY_dict=Z_XY)
        
        test_data = [Z_XY_dataset[idx] for idx in test_indices]

        # Extract the components, inner format might be numpy and cause errors
        Z = torch.cat([torch.tensor(sample["targets"]) for sample in test_data], dim=0)
        Z = push_to_device(Z, device)
        Y = torch.cat([torch.tensor(sample["inputs"]) for sample in test_data], dim=0)
        Y = push_to_device(Y, device)
        Cw_i = torch.cat([torch.tensor(sample["Cw"]) for sample in test_data], dim=0)
        Cw_i = push_to_device(Cw_i, device)

        test_data_dict = {}
        test_data_dict["Z"] = Z
        test_data_dict["Y"] = Y
        test_data_dict["Cw"] = Cw_i
        save_dataset(Z_XY=test_data_dict, filename=test_data_file)

    else:

        print("Dataset at {} already present!".format(test_data_file))
        m, n, T_test, N_test, smnr_dB_test = parse("test_sequence_m_{:d}_n_{:d}_rotnist_T_{:d}_N_{:d}_smnr_{:f}dB.pkl", test_data_file.split('/')[-1])

        test_data_dict = load_saved_dataset(filename=test_data_file)
        Z = test_data_dict["Z"]
        Y = test_data_dict["Y"]
        Cw_i = test_data_dict["Cw"]

    print("*"*100)
    print("*"*100,file=orig_stdout)
    #i_test = np.random.choice(N_test)
    print("smnr: {}dB".format(smnr_dB_test))
    print("smnr: {}dB".format(smnr_dB_test), file=orig_stdout)


    # NOTE: This is only for quick testing!! We select 2 trajectories and run 
    # all the tests using them. Uncomment them usually!
    #Y = Y[:2]
    #Z = Z[:2]

    N_test, Ty, dy = Y.shape
    N_test, Tz, dz = Z.shape

    # Get the estimate using the baseline
    H_tensor = torch.from_numpy(jacobian(h_fn, torch.randn(m,)).numpy()).type(torch.FloatTensor)
    H_tensor = torch.repeat_interleave(H_tensor.unsqueeze(0),N_test,dim=0).to(device)
    #Z_LS = torch.einsum('ijj,ikj->ikj',torch.pinverse(H_tensor),Y)
    Z_LS = torch.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Z_LS[i,j,:] = (torch.pinverse(H_tensor[i]) @ Y[i,j,:].reshape((dy, 1))).reshape((dz,))

    if "partial" in evaluation_mode:
        if "low" in evaluation_mode:
            p *= -1
            bias *= -1
        elif "high" in evaluation_mode:
            p *= 1
            bias *= 1

        #sigma_e2_dB_test = partial_corrupt((sigma_e2_dB_test), p=p, bias=bias) # 50 % corruption of the true nu_dB used for data generation
    
    # NOTE: Partial corruption code is incomplete! Needs to be fixed !!

    print("Fed to Model-based filters: ", file=orig_stdout)
    print("smnr: {}dB, delta_t: {}".format(smnr_dB_test, delta_t), file=orig_stdout)

    print("Fed to Model-based filters: ")
    print("smnr: {}dB, delta_t: {}".format(smnr_dB_test, delta_t))

    #lorenz_model.sigma_e2 = dB_to_lin(sigma_e2_dB_test)
    #lorenz_model.setStateCov(sigma_e2=dB_to_lin(sigma_e2_dB_test))

    print("Testing DANSE ...", file=orig_stdout)
    # Initialize the DANSE model in PyTorch
    _, est_dict = get_parameters(n_states=m,
                                        n_obs=n, 
                                        device=device)
    
    estimator_options = est_dict["danse"]
    estimator_options['H'] = get_H_DANSE(type_=dataset_type, n_states=n, n_obs=m) # Get the sensing matrix from the model info
    
    danse_model = DANSE(**estimator_options)

    """
    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        mu_w=lorenz_model.mu_w,
        C_w=Cw_i, # Added Cw
        batch_size=1,
        H=lorenz_model.H,#jacobian(h_fn, torch.randn(lorenz_model.n_states,)).numpy(),
        mu_x0=np.zeros((lorenz_model.n_states,)),
        C_x0=np.eye(lorenz_model.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    
    )
    """
    print("DANSE Model file: {}".format(model_file_saved))

    Z_estimated_pred = None
    Z_estimated_filtered = None
    Pk_estimated_filtered = None

    start_time_danse = timer()
    Z_estimated_pred, Pk_estimated_pred, Z_estimated_filtered, Pk_estimated_filtered = test_danse_rotnist(danse_model=danse_model, 
                                                                                                saved_model_file=model_file_saved,
                                                                                                Y=Y,
                                                                                                Cw=Cw_i,
                                                                                                device=device)  
    
    # z_samples = recreate_latent_values(Z_estimated_filtered, Pk_estimated_filtered, device)
    post_mean_z_hat = Z_estimated_filtered
    time_elapsed_danse = timer() - start_time_danse
        
    nmse_ls = nmse_loss(Z[:,:,:], Z_LS[:,0:,:])
    nmse_ls_std = nmse_loss_std(Z[:,:,:], Z_LS[:,0:,:])
    
    #nmse_ekf = nmse_loss(Z[:,:,:], X_estimated_ekf[:,:,:])
    #nmse_ekf_std = nmse_loss_std(Z[:,:,:], X_estimated_ekf[:,:,:])
    #nmse_ukf = nmse_loss(Z[:,:,:], X_estimated_ukf[:,:,:])
    #nmse_ukf_std = nmse_loss_std(Z[:,:,:], X_estimated_ukf[:,:,:])
    nmse_danse = nmse_loss(Z[:,:,:], Z_estimated_filtered[:,0:,:])
    nmse_danse_std = nmse_loss_std(Z[:,:,:], Z_estimated_filtered[:,0:,:])
    nmse_danse_pred = nmse_loss(Z[:,:,:], Z_estimated_pred[:,0:,:])
    nmse_danse_pred_std = nmse_loss_std(Z[:,:,:], Z_estimated_pred[:,0:,:])
    #nmse_knet = None #nmse_loss(Z[:,:,:], Z_estimated_filtered_knet[:,0:,:])
    #nmse_knet_std = None #nmse_loss_std(Z[:,:,:], Z_estimated_filtered_knet[:,0:,:])
    
    mse_dB_ls = mse_loss_dB(Z[:,:,:], Z_LS[:,0:,:])
    mse_dB_ls_std = mse_loss_dB_std(Z[:,:,:], Z_LS[:,0:,:])
    #mse_dB_ekf = mse_loss_dB(Z[:,:,:], X_estimated_ekf[:,:,:])
    #mse_dB_ekf_std = mse_loss_dB_std(Z[:,:,:], X_estimated_ekf[:,:,:])
    #mse_dB_ukf = mse_loss_dB(Z[:,:,:], X_estimated_ukf[:,:,:])
    #mse_dB_ukf_std = mse_loss_dB_std(Z[:,:,:], X_estimated_ukf[:,:,:])
    mse_dB_danse = mse_loss_dB(Z[:,:,:], Z_estimated_filtered[:,0:,:])
    mse_dB_danse_std = mse_loss_dB_std(Z[:,:,:], Z_estimated_filtered[:,0:,:])
    mse_dB_danse_pred = mse_loss_dB(Z[:,:,:], Z_estimated_pred[:,0:,:])
    mse_dB_danse_pred_std = mse_loss_dB_std(Z[:,:,:], Z_estimated_pred[:,0:,:])
    #mse_dB_knet = None #mse_loss_dB(Z[:,:,:], Z_estimated_filtered_knet[:,0:,:])
    #mse_dB_knet_std = None #mse_loss_dB_std(Z[:,:,:], Z_estimated_filtered_knet[:,0:,:])
    
    print("DANSE - MSE LOSS:",mse_dB_danse, "[dB]")
    print("DANSE - MSE STD:", mse_dB_danse_std, "[dB]")

    #print("KNET - MSE LOSS:", mse_dB_knet, "[dB]")
    #print("KNET - MSE STD:", mse_dB_knet_std, "[dB]")

    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std))
    #print("ekf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_ekf, nmse_ekf_std, mse_dB_ekf, mse_dB_ekf_std, time_elapsed_ekf))
    #print("ukf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_ukf, nmse_ukf_std, mse_dB_ukf, mse_dB_ukf_std, time_elapsed_ukf))
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse))
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))
    #print("knet (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_knet, nmse_knet_std, mse_dB_knet, mse_dB_knet_std, time_elapsed_knet))

    # System console print
    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std), file=orig_stdout)
    #print("ekf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_ekf, nmse_ekf_std, mse_dB_ekf, mse_dB_ekf_std, time_elapsed_ekf), file=orig_stdout)
    #print("ukf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_ukf, nmse_ukf_std, mse_dB_ukf, mse_dB_ukf_std, time_elapsed_ukf), file=orig_stdout)
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse), file=orig_stdout)
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse), file=orig_stdout)
    #print("knet (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_knet, nmse_knet_std, mse_dB_knet, mse_dB_knet_std, time_elapsed_knet), file=orig_stdout)

    """
    # Plot the result
    ifig = 0 #np.random.randint(Z.shape[0])
    
    print("Chosen trajectory index for plots: {}".format(ifig))
    print("Chosen trajectory index for plots: {}".format(ifig), file=orig_stdout)

    
    plot_3d_state_trajectory(Z=torch.squeeze(Z[ifig, 1:, :], 0).numpy(), legend='$\\mathbf{Z}^{true}$', m='b-', savefig_name="./figs/LorenzModel/{}/lorenz_x_true_sigmae2_{}dB_smnr_{}dB.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test), savefig=True)
    plot_3d_state_trajectory(Z=torch.squeeze(Z_estimated_filtered[ifig], 0).numpy(), legend='$\\hat{\mathbf{Z}}_{DANSE}$', m='k-', savefig_name="./figs/LorenzModel/{}/lorenz_x_danse_sigmae2_{}dB_smnr_{}dB.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test), savefig=True)
    plot_3d_measurment_trajectory(Y=torch.squeeze(Y[ifig, :, :], 0).numpy(), legend='$\\mathbf{y}^{true}$', m='r-', savefig_name="./figs/LorenzModel/{}/lorenz_y_true_sigmae2_{}dB_smnr_{}dB.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test), savefig=True)

    plot_state_trajectory(Z=torch.squeeze(Z[ifig,:,:],0).numpy(), 
                        #X_est_EKF=torch.squeeze(X_estimated_ekf[ifig,:,:],0).numpy(), 
                        #X_est_UKF=torch.squeeze(X_estimated_ukf[ifig,:,:],0).numpy(), 
                        X_est_DANSE=torch.squeeze(Z_estimated_filtered[ifig],0).numpy(),
                        #X_est_KNET=torch.squeeze(Z_estimated_filtered_knet[ifig], 0).numpy(),
                        savefig=True,
                        #savefig_name="./figs/LorenzModel/{}/3dPlot_sigmae2_{}dB_smnr_{}dB_knet.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test))
                        savefig_name="./figs/LorenzModel/{}/3dPlot_sigmae2_{}dB_smnr_{}dB.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test))
     
    plot_state_trajectory_w_lims(Z=torch.squeeze(Z[ifig,:,:],0).numpy(), 
                        #X_est_KF=torch.squeeze(X_estimated_kf[0,1:,:], 0).numpy(), 
                        #X_est_KF_std=np.sqrt(torch.diagonal(torch.squeeze(Pk_estimated_kf[0,1:,:,:], 0), offset=0, dim1=1,dim2=2).numpy()), 
                        #X_est_UKF=torch.squeeze(X_estimated_ukf[ifig,:,:], 0).numpy(), 
                        #X_est_UKF_std=np.sqrt(torch.diagonal(torch.squeeze(Pk_estimated_ukf[ifig,:,:,:], 0), offset=0, dim1=1,dim2=2).numpy()), 
                        X_est_DANSE=torch.squeeze(Z_estimated_filtered[ifig], 0).numpy(), 
                        X_est_DANSE_std=np.sqrt(torch.diagonal(torch.squeeze(Pk_estimated_filtered[ifig], 0), offset=0, dim1=1,dim2=2).numpy()), 
                        #X_est_DANSE_sup=torch.squeeze(Z_estimated_filtered_sup[0], 0).numpy(), 
                        #X_est_DANSE_sup_std=np.diag(torch.squeeze(Pk_estimated_filtered_sup[0,1:,:], 0).numpy()).sqrt(), 
                        #X_est_KNET=torch.squeeze(Z_estimated_filtered_knet[0], 0).numpy(), 
                        savefig=True,
                        savefig_name="./figs/LorenzModel/{}/Trajectories_sigma_e2_{}dB_smnr_{}dB.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test))
    
    plot_state_trajectory_axes(Z=torch.squeeze(Z[ifig,:,:],0).numpy(), 
                                #X_est_EKF=torch.squeeze(X_estimated_ekf[ifig,:,:],0).numpy(), 
                                #X_est_UKF=torch.squeeze(X_estimated_ukf[ifig,:,:],0).numpy(), 
                                X_est_DANSE=torch.squeeze(Z_estimated_filtered[ifig],0).numpy(), 
                                #X_est_KNET=torch.squeeze(Z_estimated_filtered_knet[ifig], 0).numpy(),
                                savefig=True,
                                #savefig_name="./figs/LorenzModel/{}/AxesWisePlot_sigmae2_{}dB_smnr_{}dB_knet.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test))
                                savefig_name="./figs/LorenzModel/{}/AxesWisePlot_sigmae2_{}dB_smnr_{}dB.pdf".format(evaluation_mode, sigma_e2_dB_test, smnr_dB_test))

    #plot_state_trajectory_axes(Z=torch.squeeze(Z,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(Z_estimated_filtered,0))
    #plot_state_trajectory(Z=torch.squeeze(Z,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(Z_estimated_filtered,0))
    
    #plt.show()
    """
    sys.stdout = orig_stdout
    return nmse_danse, nmse_danse_std, nmse_ls, nmse_ls_std, \
        mse_dB_danse, mse_dB_danse_std, mse_dB_ls, mse_dB_ls_std, \
        time_elapsed_danse, smnr_dB_test, post_mean_z_hat, test_data_dict

"""
def run_decoding(model, device, pkl_path, latent_dim, T, smnr_db):
    pkl_file = sorted(glob.glob(os.path.join(pkl_path, f"sequence_m_{latent_dim}_n_{latent_dim}_rotnist_T_{T}_*_smnr_{smnr_db}dB.pkl")))
    Z_XY_dict = load_saved_dataset(str(pkl_file[0]))
    y_array = Z_XY_dict["dataY"]
    z_array = Z_XY_dict["dataZ"]
    fp_arr = Z_XY_dict["img_fpaths"]
    decoded_z_list = list()
    decoded_y_list = list()
    reqd_fpaths = list()
    cnt = 0
    for noise_vector in y_array:
        decoded_y = decode_data(model, torch.tensor(noise_vector[0]), device)
        decoded_y_list.append(decoded_y)
        cnt += 1
        if cnt == 5:
            cnt = 0
            break
    
    for latent_vector in z_array:
        decoded_z = decode_data(model, torch.tensor(latent_vector[0]), device)
        decoded_z_list.append(decoded_z)
        cnt += 1
        if cnt == 5:
            cnt = 0
            break

    for fpath in fp_arr:
        reqd_fpaths.append(fpath[0])
        cnt += 1
        if cnt == 5:
            break
    
    return decoded_y_list, decoded_z_list, reqd_fpaths
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Input arguments related to creating a dataset for ROTNIST")
    parser.add_argument("--mode", help="Enter 'encode' or 'decode' mode", type=str, default="train")
    parser.add_argument("--model_type", help="Enter 'ae' or 'vae' model", type=str, default="ae")
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default='data/encoded_data/')
    parser.add_argument("--danse_input_path", help="Enter full path to store the danse input files", type=str, default='data/encoded_noise_data')
    parser.add_argument("--saved_model_path", help="Enter full path to save the AR/VAE model", type=str, default='models/rotnist_models/')
    parser.add_argument("--smnr_db", help="For the smnr", type=float, default=20.0)
    parser.add_argument("--latent_dim", help="For the latent dimension", type=int, default=32)


    args = parser.parse_args() 
    mode = args.mode
    model_type = args.model_type
    enc_img_output_dir = args.output_path
    saved_model_path = args.saved_model_path
    danse_input_path = args.danse_input_path
    smnr_db = args.smnr_db
    latent_dim = args.latent_dim

    # Testing parameters 
    T_test = 20
    N_test = 500
    N_train = 500
    T_train = 20
    #sigma_e2_dB_test = -10.0
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    bias = None # By default should be positive, equal to 10.0
    p = None # Keep this fixed at zero for now, equal to 0.0
    mode = 'full'
    if mode == 'low' or mode == 'high':
        evaluation_mode = 'partial_opt_{}_bias_{}_p_{}'.format(mode, bias, p)
    else:
        bias = None
        p = None
        evaluation_mode = 'full_opt_bias_{}_p_{}_modTest_danseOnly_Ttrain_{}_Ntrain_{}'.format(None, None, T_train, N_train)
        # Original: evaluation_mode = 'full_opt_bias_{}_p_{}_quicktest_Ttrain_{}_Ntrain_{}'.format(None, None, T_train, N_train)

    os.makedirs('./figs/rotnist_figs/{}'.format(evaluation_mode), exist_ok=True)

    smnr_dB_arr = np.array([0.0,10.0,20.0])
    #smnr_dB_arr = np.array([20.0])

    nmse_ls_arr = np.zeros((len(smnr_dB_arr,)))
    #nmse_ekf_arr = np.zeros((len(smnr_dB_arr,)))
    #nmse_ukf_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_danse_arr = np.zeros((len(smnr_dB_arr,)))
    #nmse_knet_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_ls_std_arr = np.zeros((len(smnr_dB_arr,)))
    #nmse_ekf_std_arr = np.zeros((len(smnr_dB_arr,)))
    #nmse_ukf_std_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_danse_std_arr = np.zeros((len(smnr_dB_arr,)))
    #nmse_knet_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_ls_dB_arr = np.zeros((len(smnr_dB_arr,)))
    #mse_ekf_dB_arr = np.zeros((len(smnr_dB_arr,)))
    #mse_ukf_dB_arr = np.zeros((len(smnr_dB_arr,)))
    mse_danse_dB_arr = np.zeros((len(smnr_dB_arr,)))
    #mse_knet_dB_arr = np.zeros((len(smnr_dB_arr,)))
    mse_ls_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    #mse_ekf_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    #mse_ukf_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_danse_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    #mse_knet_dB_std_arr = np.zeros((len(smnr_dB_arr,)))
    #t_ekf_arr = np.zeros((len(smnr_dB_arr,)))
    #t_ukf_arr = np.zeros((len(smnr_dB_arr,)))
    t_danse_arr = np.zeros((len(smnr_dB_arr,)))
    #t_knet_arr = np.zeros((len(smnr_dB_arr,)))
    snr_arr = np.zeros((len(smnr_dB_arr,)))

    model_file_saved_dict = {}
    #model_file_saved_dict_knet = {}

    for smnr_dB in smnr_dB_arr:
        model_file_saved_dict["{}dB".format(smnr_dB)] = glob.glob(r"./models/*rotnist_danse_opt_*n_32_T_{}_N_{}_smnr_{}dB*/*best*".format(T_train, N_train, smnr_dB))[-1]

    test_data_file_dict = {}

    for smnr_dB in smnr_dB_arr:
        test_data_file_dict["{}dB".format(smnr_dB)] = "./data/encoded_noise_data/test_sequence_m_32_n_32_rotnist_T_{}_N_{}_smnr_{}dB.pkl".format(T_test, N_test, smnr_dB)
    
    print("*"*100)
    print(model_file_saved_dict)
    print("*"*100)
    print(test_data_file_dict)
    
    #test_logfile = "./log/Lorenz_test_{}_T_{}_N_{}_w_knet_modTest.log".format(evaluation_mode, T_test, N_test)
    #test_jsonfile = "./log/Lorenz_test_{}_T_{}_N_{}_w_knet_modTest.json".format(evaluation_mode, T_test, N_test)

    test_logfile = "./log/Rotnist_test_{}_T_{}_N_{}_log.log".format(evaluation_mode, T_test, N_test) # Original
    test_jsonfile = "./log/Rotnist_test_{}_T_{}_N_{}_results.json".format(evaluation_mode, T_test, N_test) # Original

    saved_model_path="./models/rotnist_models"
    model_type = 'ae'
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = torch.load(os.path.join(saved_model_path, f"{model_type}_model")).to(device)

    recon_img_dict = dict()

    for i, smnr_dB in enumerate(smnr_dB_arr):
        
        model_file_saved_i = model_file_saved_dict['{}dB'.format(smnr_dB)].replace("\\", "/")
        test_data_file_i = test_data_file_dict['{}dB'.format(smnr_dB)]
        #model_file_saved_knet_i = model_file_saved_dict_knet['{}dB'.format(smnr_dB)]

        nmse_danse_i, nmse_danse_i_std, nmse_ls_i, nmse_ls_i_std, \
            mse_dB_danse_i, mse_dB_danse_std_i, mse_dB_ls_i, mse_dB_ls_std_i, \
            time_elapsed_danse_i, smnr_dB_i, post_mean_z_hat_i, test_data_dict_i = test_rotnist(device=device, 
            model_file_saved=model_file_saved_i, test_data_file=test_data_file_i, test_logfile=test_logfile, 
            evaluation_mode=evaluation_mode, bias=bias, p=p)

# Run decoding
#-------------------------------------------------------------------------------

        n_list = []   
        for num_sample in range(post_mean_z_hat_i.shape[0]):
            t_list = []
            for t_sample in range(post_mean_z_hat_i.shape[1]):
                elem = post_mean_z_hat_i[num_sample, t_sample]
                x_hat = decode_data(model, elem, device) # x_hat = 784
                t_list.append(x_hat) 
            n_list.append(t_list) # n_list len = 50
        
        recon_img_dict[str(smnr_dB)] = n_list
        recon_img_dict["dataZ"] = test_data_dict_i["Z"]
            
#-------------------------------------------------------------------------------

        # Store the NMSE values and std devs of the NMSE values
        nmse_ls_arr[i] = nmse_ls_i.item()
        #nmse_ekf_arr[i] = nmse_ekf_i.numpy().item()
        #nmse_ukf_arr[i] = nmse_ukf_i.numpy().item()
        nmse_danse_arr[i] = nmse_danse_i.item()
        #nmse_knet_arr[i] = nmse_knet_i.numpy().item()
        nmse_ls_std_arr[i] = nmse_ls_i_std.item()
        #nmse_ekf_std_arr[i] = nmse_ekf_i_std.numpy().item()
        #nmse_ukf_std_arr[i] = nmse_ukf_i_std.numpy().item()
        nmse_danse_std_arr[i] = nmse_danse_i_std.item()
        #nmse_knet_std_arr[i] = nmse_knet_std_i.numpy().item()
        
        # Store the MSE values and std devs of the MSE values (in dB)
        mse_ls_dB_arr[i] = mse_dB_ls_i.item()
        #mse_ekf_dB_arr[i] = mse_dB_ekf_i.numpy().item()
        #mse_ukf_dB_arr[i] = mse_dB_ukf_i.numpy().item()
        mse_danse_dB_arr[i] = mse_dB_danse_i.item()
        #mse_knet_dB_arr[i] = mse_dB_knet_i.numpy().item()
        mse_ls_dB_std_arr[i] = mse_dB_ls_std_i.item()
        #mse_ekf_dB_std_arr[i] = mse_dB_ekf_std_i.numpy().item()
        #mse_ukf_dB_std_arr[i] = mse_dB_ukf_std_i.numpy().item()
        mse_danse_dB_std_arr[i] = mse_dB_danse_std_i.item()
        #mse_knet_dB_std_arr[i] = mse_dB_knet_std_i.numpy().item()

        # Store the inference times
        #t_ekf_arr[i] = time_elapsed_ekf_i
        #t_ukf_arr[i] = time_elapsed_ukf_i
        t_danse_arr[i] = time_elapsed_danse_i
        #t_knet_arr[i] = time_elapsed_knet_i
    
    with torch.no_grad():
        with PdfPages('reconstructed_images_danse.pdf') as pdf:
            keys = ['dataZ', '0.0', '10.0', '20.0']
            fig, axs = plt.subplots(len(keys), len(n_list[0]), figsize=(20, 4 * len(keys)))

            for i, key in enumerate(keys):
                if key == "dataZ":
                    dataZ_list = []
                    for num_sample in range(recon_img_dict["dataZ"].shape[0]):
                        t_list = []
                        for t_sample in range(recon_img_dict["dataZ"].shape[1]):
                            elem = recon_img_dict["dataZ"][num_sample, t_sample]
                            x_hat = decode_data(model, elem, device)
                            t_list.append(x_hat)
                        dataZ_list.append(t_list)
                    
                    for k in range(len(dataZ_list[0])):
                        decoded_img = dataZ_list[0][k]
                        axs[i, k].imshow(decoded_img.cpu().view(28, 28), cmap='gray')
                        axs[i, k].set_title(f'dataZ {k}')
                        axs[i, k].axis('off')
                else:
                    for k in range(len(recon_img_dict[key][0])):
                        reconstructed_image = recon_img_dict[key][0][k].cpu().view(28, 28)
                        axs[i, k].imshow(reconstructed_image, cmap='gray')
                        axs[i, k].set_title(f'{key} dB {k}')
                        axs[i, k].axis('off')

            pdf.savefig(fig)
            plt.close(fig)

    print("Saved reconstructed images to PDF.")
    
    test_stats = {}
    #test_stats['UKF_mean_nmse'] = nmse_ukf_arr
    #test_stats['EKF_mean_nmse'] = nmse_ekf_arr
    test_stats['DANSE_mean_nmse'] = nmse_danse_arr
    #test_stats['KNET_mean_nmse'] = nmse_knet_arr
    #test_stats['UKF_std_nmse'] = nmse_ukf_std_arr
    #test_stats['EKF_std_nmse'] = nmse_ekf_std_arr
    test_stats['DANSE_std_nmse'] = nmse_danse_std_arr
    #test_stats['KNET_std_nmse'] = nmse_knet_std_arr
    test_stats['LS_mean_nmse'] = nmse_ls_arr
    test_stats['LS_std_nmse'] = nmse_ls_std_arr

    #test_stats['EKF_mean_mse'] = mse_ekf_dB_arr
    #test_stats['UKF_mean_mse'] = mse_ukf_dB_arr
    test_stats['DANSE_mean_mse'] = mse_danse_dB_arr
    #test_stats['KNET_mean_mse'] = mse_knet_dB_arr
    #test_stats['EKF_std_mse'] = mse_ekf_dB_std_arr
    #test_stats['UKF_std_mse'] = mse_ukf_dB_std_arr
    test_stats['DANSE_std_mse'] = mse_danse_dB_std_arr
    #test_stats['KNET_std_mse'] = mse_knet_dB_std_arr
    test_stats['LS_mean_mse'] = mse_ls_dB_arr
    test_stats['LS_std_mse'] = mse_ls_dB_std_arr

    #test_stats['UKF_time'] = t_ukf_arr
    #test_stats['EKF_time'] = t_ekf_arr
    test_stats['DANSE_time'] = t_danse_arr
    #test_stats['KNET_time'] = t_knet_arr
    test_stats['SMNR'] = smnr_dB_arr
    
    with open(test_jsonfile, 'w') as f:
        f.write(json.dumps(test_stats, cls=NDArrayEncoder, indent=2))

    # Plotting the NMSE Curve
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.errorbar(smnr_dB_arr, nmse_ls_arr, fmt='gp-.', yerr=nmse_ls_std_arr,  linewidth=1.5, label="LS")
    #plt.errorbar(smnr_dB_arr, nmse_ekf_arr, fmt='rd--',  yerr=nmse_ekf_std_arr, linewidth=1.5, label="EKF")
    #plt.errorbar(smnr_dB_arr, nmse_ukf_arr, fmt='ko-',  yerr=nmse_ukf_std_arr, linewidth=1.5, label="UKF")
    plt.errorbar(smnr_dB_arr, nmse_danse_arr, fmt='b*-', yerr=nmse_danse_std_arr, linewidth=2.0, label="DANSE")
    #plt.errorbar(smnr_dB_arr, nmse_knet_arr, fmt='ys-', yerr=nmse_knet_std_arr,  linewidth=1.0, label="KalmanNet")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.subplot(212)
    tikzplotlib.save('./figs/rotnist_figs/{}/NMSE_vs_SMNR_Rotnist.tex'.format(evaluation_mode))
    plt.savefig('./figs/rotnist_figs/{}/NMSE_vs_SMNR_Rotnist.pdf'.format(evaluation_mode))

    # Plotting the Time-elapsed Curve
    plt.figure()
    #plt.subplot(211)
    #plt.plot(smnr_dB_arr, t_ekf_arr, 'rd--', linewidth=1.5, label="EKF")
    #plt.plot(smnr_dB_arr, t_ukf_arr, 'ks--', linewidth=1.5, label="UKF")
    plt.plot(smnr_dB_arr, t_danse_arr, 'bo-', linewidth=2.0, label="DANSE")
    #plt.plot(smnr_dB_arr, t_knet_arr, 'ys-', linewidth=1.0, label="KalmanNet")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('Inference time (in s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #tikzplotlib.save('./figs/LorenzModel/{}/InferTime_vs_SMNR_Lorenz_w_knet.tex'.format(evaluation_mode))
    #plt.savefig('./figs/LorenzModel/{}/InferTime_vs_SMNR_Lorenz_w_knet.pdf'.format(evaluation_mode))
    tikzplotlib.save('./figs/rotnist_figs/{}/InferTime_vs_SMNR_Rotnist.tex'.format(evaluation_mode)) # Original
    plt.savefig('./figs/rotnist_figs/{}/InferTime_vs_SMNR_Rotnist.pdf'.format(evaluation_mode)) # Original



    # Plotting the MSE Curve
    plt.figure()
    plt.errorbar(smnr_dB_arr, mse_ls_dB_arr, fmt='gp-.', yerr=mse_ls_dB_std_arr,  linewidth=1.5, label="LS")
    #plt.errorbar(smnr_dB_arr, mse_ekf_dB_arr, fmt='rd--',  yerr=mse_ekf_dB_std_arr, linewidth=1.5, label="EKF")
    #plt.errorbar(smnr_dB_arr, mse_ukf_dB_arr, fmt='ko-',  yerr=mse_ukf_dB_std_arr, linewidth=1.5, label="UKF")
    plt.errorbar(smnr_dB_arr, mse_danse_dB_arr, fmt='b*-', yerr=mse_danse_dB_std_arr, linewidth=2.0, label="DANSE")
    #plt.errorbar(smnr_dB_arr, mse_knet_dB_arr, fmt='ys-', yerr=mse_knet_dB_std_arr,  linewidth=1.0, label="KalmanNet")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('MSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.subplot(212)
    #tikzplotlib.save('./figs/LorenzModel/{}/MSE_vs_SMNR_Lorenz_w_knet.tex'.format(evaluation_mode))
    #plt.savefig('./figs/LorenzModel/{}/MSE_vs_SMNR_Lorenz_w_knet.pdf'.format(evaluation_mode))
    tikzplotlib.save('./figs/rotnist_figs/{}/MSE_vs_SMNR_Rotnist.tex'.format(evaluation_mode)) # Original
    plt.savefig('./figs/rotnist_figs/{}/MSE_vs_SMNR_Rotnist.pdf'.format(evaluation_mode)) # Original

    #plt.show()
