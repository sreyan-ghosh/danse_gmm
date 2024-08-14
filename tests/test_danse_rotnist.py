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
from PIL import Image
import json
import tikzplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import for decoding:
from src.rotnist_enc_dec import AE, decode_data

from utils.plot_functions import *
from utils.utils_rotnist import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss, \
    mse_loss_dB, load_saved_dataset, save_dataset, nmse_loss_std, mse_loss_dB_std, NDArrayEncoder, partial_corrupt, \
    get_dataloaders, Series_Dataset, push_to_device, psnr_loss, psnr_loss_std, ssim_loss, ssim_loss_std
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
        
        data_par_dir = "data/encoded_noise_data/test_data"
        datafile = os.path.join(data_par_dir, f"sequence_m_{m}_n_{n}_rotnist_T_{T_test}_N_{N_test}_smnr_{smnr_dB_test}dB.pkl")
        Z_XY = load_saved_dataset(filename=datafile)

        # Extract the components, inner format might be numpy and cause errors
        Z = torch.from_numpy(Z_XY["dataZ"]).to(device)
        Y = torch.from_numpy(Z_XY["dataY"]).to(device)
        Cw_i = torch.from_numpy(Z_XY["dataCw"]).to(device)
        fpaths = Z_XY["img_fpaths"]

        test_data_dict = {}
        test_data_dict["Z"] = Z
        test_data_dict["Y"] = Y
        test_data_dict["Cw"] = Cw_i
        test_data_dict["img_paths"] =  fpaths
        save_dataset(Z_XY=test_data_dict, filename=test_data_file)

    else:

        print("Dataset at {} already present!".format(test_data_file))
        m, n, T_test, N_test, smnr_dB_test = parse("test_sequence_m_{:d}_n_{:d}_rotnist_T_{:d}_N_{:d}_smnr_{:f}dB.pkl", test_data_file.split('/')[-1])

        test_data_dict = load_saved_dataset(filename=test_data_file)
        Z = test_data_dict["Z"]
        Y = test_data_dict["Y"]
        Cw_i = test_data_dict["Cw"]
        fpaths = test_data_dict["img_paths"]

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

    print("Fed to Model-based filters: ", file=orig_stdout)
    print("smnr: {}dB, delta_t: {}".format(smnr_dB_test, delta_t), file=orig_stdout)

    print("Fed to Model-based filters: ")
    print("smnr: {}dB, delta_t: {}".format(smnr_dB_test, delta_t))

    print("Testing DANSE ...", file=orig_stdout)
    # Initialize the DANSE model in PyTorch
    _, est_dict = get_parameters(n_states=m,
                                        n_obs=n, 
                                        device=device)
    
    estimator_options = est_dict["danse"]
    estimator_options['H'] = get_H_DANSE(type_=dataset_type, n_states=n, n_obs=m) # Get the sensing matrix from the model info
    
    danse_model = DANSE(**estimator_options)

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

    # X dims = (100,20,784)
    X = []
    fpaths = np.asarray(fpaths)
    for i in range(fpaths.shape[0]):
        row = []
        for j in range(fpaths.shape[1]):
            img = Image.open(fpaths[i, j])
            img = img.convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img).reshape(-1) / 255.0
            img_tensor = torch.tensor(img_array, dtype=torch.float32).to(device)
            row.append(img_tensor)
        X.append(torch.stack(row).to(device))

    X = torch.stack(X).to(device)

    # Run decoding
#-------------------------------------------------------------------------------

    xhat_danse = []   
    for num_sample in range(post_mean_z_hat.shape[0]):
        t_list = []
        for t_sample in range(post_mean_z_hat.shape[1]):
            elem = post_mean_z_hat[num_sample, t_sample]
            x_hat = decode_data(model, elem, device) # x_hat = 784
            t_list.append(x_hat) 
        xhat_danse.append(t_list) # xhat_arr len = 100
    
    xhat_danse = torch.stack([torch.stack(t_list) for t_list in xhat_danse])

    xhat_ls = []   
    for num_sample in range(Z_LS.shape[0]):
        t_list = []
        for t_sample in range(Z_LS.shape[1]):
            elem = Z_LS[num_sample, t_sample]
            x_hat = decode_data(model, elem, device) 
            t_list.append(x_hat) 
        xhat_ls.append(t_list) 
    xhat_ls = torch.stack([torch.stack(t_list) for t_list in xhat_ls])

#-----------------------------------------------------------------------------------
    # NMSE
    nmse_ls_z = nmse_loss(Z, Z_LS)
    nmse_ls_z_std = nmse_loss_std(Z, Z_LS)
    nmse_ls_x = nmse_loss(X, xhat_ls)
    nmse_ls_x_std = nmse_loss_std(X, xhat_ls)
    
    nmse_danse_z = nmse_loss(Z, post_mean_z_hat)
    nmse_danse_z_std = nmse_loss_std(Z, post_mean_z_hat)
    nmse_danse_x = nmse_loss(X, xhat_danse)
    nmse_danse_x_std = nmse_loss_std(X, xhat_danse)

    # MSE
    mse_dB_ls_z = mse_loss_dB(Z, Z_LS)
    mse_dB_ls_z_std = mse_loss_dB_std(Z, Z_LS)
    mse_dB_ls_x = mse_loss_dB(X, xhat_ls)
    mse_dB_ls_x_std = mse_loss_dB_std(X, xhat_ls)

    mse_dB_danse_z = mse_loss_dB(Z, post_mean_z_hat)
    mse_dB_danse_z_std = mse_loss_dB_std(Z, post_mean_z_hat)
    mse_dB_danse_x = mse_loss_dB(X, xhat_danse)
    mse_dB_danse_x_std = mse_loss_dB_std(X, xhat_danse)
    
    # Log file print
    print("DANSE - MSE LOSS:",mse_dB_danse_x, "[dB]")
    print("DANSE - MSE STD:", mse_dB_danse_x_std, "[dB]")

    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls_x, nmse_ls_x_std, mse_dB_ls_x, mse_dB_ls_x_std))
    # print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred_x, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse))
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_x, nmse_danse_x_std, mse_dB_danse_x, mse_dB_danse_x_std, time_elapsed_danse))

    # System console print
    sys.stdout = orig_stdout
    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls_x, nmse_ls_x_std, mse_dB_ls_x, mse_dB_ls_x_std))
    # print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred_x, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse))
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_x, nmse_danse_x_std, mse_dB_danse_x, mse_dB_danse_x_std, time_elapsed_danse))

    return nmse_danse_x, nmse_danse_x_std, nmse_ls_x, nmse_ls_x_std, nmse_ls_z, nmse_ls_z_std, nmse_danse_z, nmse_danse_z_std, \
        mse_dB_danse_x, mse_dB_danse_x_std, mse_dB_ls_x, mse_dB_ls_x_std, mse_dB_ls_z, mse_dB_ls_z_std, mse_dB_danse_z, mse_dB_danse_z_std,\
        time_elapsed_danse, test_data_dict, xhat_ls, xhat_danse, X


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
    T_test = 60
    N_test = 100
    N_train = 500
    T_train = 60
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

    smnr_dB_arr = np.array([-5.0, 0.0, 5.0, 10.0])

    nmse_ls_x_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_ls_z_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_ls_x_std_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_ls_z_std_arr = np.zeros((len(smnr_dB_arr,)))

    nmse_danse_x_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_danse_z_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_danse_x_std_arr = np.zeros((len(smnr_dB_arr,)))
    nmse_danse_z_std_arr = np.zeros((len(smnr_dB_arr,)))

    psnr_ls_arr = np.zeros((len(smnr_dB_arr,)))
    psnr_ls_std_arr = np.zeros((len(smnr_dB_arr,)))
    psnr_danse_arr = np.zeros((len(smnr_dB_arr,)))
    psnr_danse_std_arr = np.zeros((len(smnr_dB_arr,)))
    
    ssim_danse_arr = np.zeros((len(smnr_dB_arr,)))
    ssim_danse_std_arr = np.zeros((len(smnr_dB_arr,)))
    ssim_ls_arr = np.zeros((len(smnr_dB_arr,)))
    ssim_ls_std_arr = np.zeros((len(smnr_dB_arr,)))

    mse_ls_x_arr = np.zeros((len(smnr_dB_arr,)))
    mse_ls_z_arr = np.zeros((len(smnr_dB_arr,)))
    mse_ls_x_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_ls_z_std_arr = np.zeros((len(smnr_dB_arr,)))

    mse_danse_x_arr = np.zeros((len(smnr_dB_arr,)))
    mse_danse_z_arr = np.zeros((len(smnr_dB_arr,)))
    mse_danse_x_std_arr = np.zeros((len(smnr_dB_arr,)))
    mse_danse_z_std_arr = np.zeros((len(smnr_dB_arr,)))

    t_danse_arr = np.zeros((len(smnr_dB_arr,)))
    snr_arr = np.zeros((len(smnr_dB_arr,)))

    model_file_saved_dict = {}

    for smnr_dB in smnr_dB_arr:
        model_file_saved_dict["{}dB".format(smnr_dB)] = glob.glob(r"./models/*rotnist_danse_opt_*n_16_T_{}_N_{}_smnr_{}dB*/*best*".format(T_train, N_train, smnr_dB))[-1]

    test_data_file_dict = {}
    # Rename to test_data directory
    for smnr_dB in smnr_dB_arr:
        test_data_file_dict["{}dB".format(smnr_dB)] = "./data/encoded_noise_data/test_data/test_sequence_m_16_n_16_rotnist_T_{}_N_{}_smnr_{}dB.pkl".format(T_test, N_test, smnr_dB)
    
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


    for i, smnr_dB in enumerate(smnr_dB_arr):
        
        model_file_saved_i = model_file_saved_dict['{}dB'.format(smnr_dB)].replace("\\", "/")
        test_data_file_i = test_data_file_dict['{}dB'.format(smnr_dB)]
        #model_file_saved_knet_i = model_file_saved_dict_knet['{}dB'.format(smnr_dB)]

        nmse_danse_x, nmse_danse_x_std, nmse_ls_x, nmse_ls_x_std, nmse_ls_z, nmse_ls_z_std, nmse_danse_z, nmse_danse_z_std, \
        mse_dB_danse_x, mse_dB_danse_x_std, mse_dB_ls_x, mse_dB_ls_x_std, mse_dB_ls_z, mse_dB_ls_z_std, mse_dB_danse_z, mse_dB_danse_z_std,\
        time_elapsed_danse, test_data_dict_i, xhat_ls_i, xhat_danse_i, X = test_rotnist(device=device, 
        model_file_saved=model_file_saved_i, test_data_file=test_data_file_i, test_logfile=test_logfile, 
        evaluation_mode=evaluation_mode, bias=bias, p=p)

        recon_img_dict = dict()
        recon_img_dict["dataZ"] = test_data_dict_i["Z"]
        recon_img_dict[f"DANSE {str(smnr_dB)}"] = xhat_danse_i
        recon_img_dict[f"LS {str(smnr_dB)}"] = xhat_ls_i

        # PSNR and SSIM
        #-------------------------------------------------------------------------------
        psnr_danse_i = psnr_loss(X, xhat_danse_i)
        psnr_danse_arr[i] = psnr_danse_i
        psnr_danse_std_i = psnr_loss_std(X, xhat_danse_i)
        psnr_danse_std_arr[i] = psnr_danse_std_i

        ssim_mean_danse_i, ssim_values_danse_i = ssim_loss(X, xhat_danse_i)
        ssim_danse_arr[i] = ssim_mean_danse_i
        ssim_danse_std_i = ssim_loss_std(X, xhat_danse_i)
        ssim_danse_std_arr[i] = ssim_danse_std_i
        
        psnr_ls_i = psnr_loss(X, xhat_ls_i)
        psnr_ls_arr[i] = psnr_ls_i
        psnr_ls_std_i = psnr_loss_std(X, xhat_ls_i)
        psnr_ls_std_arr[i] = psnr_ls_std_i

        ssim_mean_ls_i, ssim_values_ls_i = ssim_loss(X, xhat_ls_i)
        ssim_ls_arr[i] = ssim_mean_ls_i
        ssim_ls_std_i = ssim_loss_std(X, xhat_ls_i)
        ssim_ls_std_arr[i] = ssim_ls_std_i

        #-------------------------------------------------------------------------------
        # Store the NMSE values and std devs of the NMSE values
        nmse_ls_z_arr[i] = nmse_ls_z.item()
        nmse_ls_z_std_arr[i] = nmse_ls_z_std.item()
        nmse_ls_x_arr[i] = nmse_ls_x.item()
        nmse_ls_x_std_arr[i] = nmse_ls_x_std.item()
        
        nmse_danse_z_arr[i] = nmse_danse_z.item()
        nmse_danse_z_std_arr[i] = nmse_danse_z_std.item()
        nmse_danse_x_arr[i] = nmse_danse_x.item()
        nmse_danse_x_std_arr[i] = nmse_danse_x_std.item()
        
        # Store the MSE values and std devs of the MSE values (in dB)
        mse_ls_z_arr[i] = mse_dB_ls_z.item()
        mse_ls_z_std_arr[i] = mse_dB_ls_z_std.item()
        mse_ls_x_arr[i] = mse_dB_ls_x.item()
        mse_ls_x_std_arr[i] = mse_dB_ls_x_std.item()
    
        mse_danse_z_arr[i] = mse_dB_danse_z.item()
        mse_danse_z_std_arr[i] = mse_dB_danse_z_std.item()
        mse_danse_x_arr[i] = mse_dB_danse_x.item()
        mse_danse_x_std_arr[i] = mse_dB_danse_x_std.item()

        # Store the inference times
        t_danse_arr[i] = time_elapsed_danse
    
    # Need to change plotting logic to accommodate origX, reconZ, danseXhat, lsXhat in one file for each SMNR
        with torch.no_grad():
            savepath = f'./figs/rotnist_figs/{evaluation_mode}/reconstructed_images_danse_smnr_{int(smnr_dB)}dB.pdf'
            with PdfPages(savepath) as pdf:
                keys = recon_img_dict.keys()
                num_cols = 60  # Number of columns to display = setting to show 10 images
                num_rows = 4  # Number of rows, showing origX, reconZ, danseXhat, lsXhat in each pdf

                fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

                # Add X images at the top
                for k in range(num_cols):
                    original_img = X[61, k].cpu().view(28, 28)
                    axs[0, k].imshow(original_img, cmap='gray')
                    axs[0, k].set_title(f'Original X {k}')
                    axs[0, k].axis('off')

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

                        for k in range(num_cols):
                            decoded_img = dataZ_list[61][k]
                            axs[i + 1, k].imshow(decoded_img.cpu().view(28, 28), cmap='gray')
                            axs[i + 1, k].set_title(f'reconZ {k}')
                            axs[i + 1, k].axis('off')

                    else:
                        for k in range(num_cols):
                            if key in [f"DANSE {str(smnr_dB)}", f"LS {str(smnr_dB)}"]:
                                reconstructed_image = recon_img_dict[key][61][k].cpu().view(28, 28)
                                axs[i + 1, k].imshow(reconstructed_image, cmap='gray')
                                axs[i + 1, k].set_title(f'{key} dB {k}')
                                axs[i + 1, k].axis('off')

                pdf.savefig(fig)
                plt.close(fig)

    print("Saved reconstructed images to PDF.")
    
    test_stats = {}

    test_stats['DANSE_mean_nmse_x'] = nmse_danse_x_arr
    test_stats['DANSE_mean_nmse_z'] = nmse_danse_z_arr
    test_stats['LS_mean_nmse_x'] = nmse_ls_x_arr
    test_stats['LS_mean_nmse_z'] = nmse_ls_z_arr
    
    test_stats['DANSE_std_nmse_x'] = nmse_danse_x_std_arr
    test_stats['DANSE_std_nmse_z'] = nmse_danse_z_std_arr
    test_stats['LS_std_nmse_x'] = nmse_ls_x_std_arr
    test_stats['LS_std_nmse_z'] = nmse_ls_z_std_arr

    test_stats["DANSE_mean_psnr"] = psnr_danse_arr
    test_stats["DANSE_std_psnr"] = psnr_danse_std_arr
    test_stats["LS_mean_psnr"] = psnr_ls_arr
    test_stats["LS_std_psnr"] = psnr_ls_std_arr
    
    test_stats["DANSE_mean_ssim"] = ssim_danse_arr
    test_stats["DANSE_std_ssim"] = ssim_danse_std_arr
    test_stats["LS_mean_ssim"] = ssim_ls_arr
    test_stats["LS_std_ssim"] = ssim_ls_std_arr

    test_stats['DANSE_mean_mse_x'] = mse_danse_x_arr
    test_stats['DANSE_mean_mse_z'] = mse_danse_z_arr
    test_stats['LS_mean_mse_x'] = mse_ls_x_arr
    test_stats['LS_mean_mse_z'] = mse_ls_z_arr
    
    test_stats['DANSE_std_mse_x'] = mse_danse_x_std_arr
    test_stats['DANSE_std_mse_z'] = mse_danse_z_std_arr    
    test_stats['LS_std_mse_x'] = mse_ls_x_std_arr    
    test_stats['LS_std_mse_z'] = mse_ls_z_std_arr
    
    test_stats['DANSE_time'] = t_danse_arr

    test_stats['SMNR'] = smnr_dB_arr
    
    with open(test_jsonfile, 'w') as f:
        f.write(json.dumps(test_stats, cls=NDArrayEncoder, indent=2))

    # Plotting the NMSE Curve for X
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.errorbar(smnr_dB_arr, nmse_ls_x_arr, fmt='gp-.', yerr=nmse_ls_x_std_arr,  linewidth=1.5, label="LS")
    plt.errorbar(smnr_dB_arr, nmse_danse_x_arr, fmt='b*-', yerr=nmse_danse_x_std_arr, linewidth=2.0, label="DANSE")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    tikzplotlib.save('./figs/rotnist_figs/{}/NMSE_vs_SMNR_X_Rotnist.tex'.format(evaluation_mode))
    plt.savefig('./figs/rotnist_figs/{}/NMSE_vs_SMNR_X_Rotnist.pdf'.format(evaluation_mode))
    
    # Plotting the NMSE Curve for Z
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.errorbar(smnr_dB_arr, nmse_ls_z_arr, fmt='gp-.', yerr=nmse_ls_z_std_arr,  linewidth=1.5, label="LS")
    plt.errorbar(smnr_dB_arr, nmse_danse_z_arr, fmt='b*-', yerr=nmse_danse_z_std_arr, linewidth=2.0, label="DANSE")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.subplot(212)
    tikzplotlib.save('./figs/rotnist_figs/{}/NMSE_vs_SMNR_Z_Rotnist.tex'.format(evaluation_mode))
    plt.savefig('./figs/rotnist_figs/{}/NMSE_vs_SMNR_Z_Rotnist.pdf'.format(evaluation_mode))

    # Plotting the PSNR Curve
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.errorbar(smnr_dB_arr, psnr_ls_arr, fmt='gp-.', yerr=psnr_ls_std_arr,  linewidth=1.5, label="LS")
    plt.errorbar(smnr_dB_arr, psnr_danse_arr, fmt='b*-', yerr=psnr_danse_std_arr, linewidth=2.0, label="DANSE")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('PSNR (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.subplot(212)
    tikzplotlib.save('./figs/rotnist_figs/{}/PSNR_vs_SMNR_Rotnist.tex'.format(evaluation_mode))
    plt.savefig('./figs/rotnist_figs/{}/PSNR_vs_SMNR_Rotnist.pdf'.format(evaluation_mode))

    # Plotting the SSIM Curve
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.errorbar(smnr_dB_arr, ssim_ls_arr, fmt='gp-.', yerr=ssim_ls_std_arr,  linewidth=1.5, label="LS")
    plt.errorbar(smnr_dB_arr, ssim_danse_arr, fmt='b*-', yerr=ssim_danse_std_arr, linewidth=2.0, label="DANSE")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.subplot(212)
    tikzplotlib.save('./figs/rotnist_figs/{}/SSIM_vs_SMNR_Rotnist.tex'.format(evaluation_mode))
    plt.savefig('./figs/rotnist_figs/{}/SSIM_vs_SMNR_Rotnist.pdf'.format(evaluation_mode))

    # Plotting the Time-elapsed Curve
    plt.figure()
    plt.plot(smnr_dB_arr, t_danse_arr, 'bo-', linewidth=2.0, label="DANSE")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('Inference time (in s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    tikzplotlib.save('./figs/rotnist_figs/{}/InferTime_vs_SMNR_Rotnist.tex'.format(evaluation_mode)) # Original
    plt.savefig('./figs/rotnist_figs/{}/InferTime_vs_SMNR_Rotnist.pdf'.format(evaluation_mode)) # Original

    # Plotting the MSE Curve for X
    plt.figure()
    plt.errorbar(smnr_dB_arr, mse_ls_x_arr, fmt='gp-.', yerr=mse_ls_x_std_arr,  linewidth=1.5, label="LS")
    plt.errorbar(smnr_dB_arr, mse_danse_x_arr, fmt='b*-', yerr=mse_danse_x_std_arr, linewidth=2.0, label="DANSE")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('MSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    tikzplotlib.save('./figs/rotnist_figs/{}/MSE_vs_SMNR_X_Rotnist.tex'.format(evaluation_mode)) # Original
    plt.savefig('./figs/rotnist_figs/{}/MSE_vs_SMNR_X_Rotnist.pdf'.format(evaluation_mode)) # Original

    # Plotting the MSE Curve for Z
    plt.figure()
    plt.errorbar(smnr_dB_arr, mse_ls_z_arr, fmt='gp-.', yerr=mse_ls_z_std_arr,  linewidth=1.5, label="LS")
    plt.errorbar(smnr_dB_arr, mse_danse_z_arr, fmt='b*-', yerr=mse_danse_z_std_arr, linewidth=2.0, label="DANSE")
    plt.xlabel('SMNR (in dB)')
    plt.ylabel('MSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    tikzplotlib.save('./figs/rotnist_figs/{}/MSE_vs_SMNR_Z_Rotnist.tex'.format(evaluation_mode)) # Original
    plt.savefig('./figs/rotnist_figs/{}/MSE_vs_SMNR_Z_Rotnist.pdf'.format(evaluation_mode)) # Original
