#####################################################
# Creators: Anubhab Ghosh, Antoine Honoré, Sreyan Ghosh, Kasper Malm
# Feb 2023, Updated: Jul 2024
#####################################################
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim, distributions
from timeit import default_timer as timer
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import copy
import math
from utils.utils_rotnist import compute_log_prob_normal, create_diag, compute_inverse, count_params, ConvergenceMonitor
#from utils.plot_functions import plot_state_trajectory, plot_state_trajectory_axes
import torch.nn.functional as F
from src.rnn_gmm import RNN_model

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    return None

def push_model(nets, device='cpu'):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    nets = nets.to(device=device)
    return nets

class DANSE_GMM(nn.Module):

    def __init__(self, n_states, n_obs, mu_w, C_w, H, mu_x0, C_x0, batch_size, rnn_type, num_mixtures, rnn_params_dict, device='cpu'):
        super(DANSE_GMM, self).__init__()

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        

        # Initialize the paramters of the state estimator
        self.n_states = n_states
        self.n_obs = n_obs
        
        # Initializing the parameters of the initial state
        self.mu_x0 = self.push_to_device(mu_x0)
        self.C_x0 = self.push_to_device(C_x0)

        # Initializing the parameters of the measurement noise
        self.mu_w = self.push_to_device(mu_w)
        self.C_w = None

        # Initialize the observation model matrix 
        self.H = self.push_to_device(H)
        
        self.batch_size = batch_size

        # Initialize RNN type
        self.rnn_type = rnn_type

        # GMM Update: Added num_mix parameter for DANSE and RNN
        # Intialize number of mixtures for GMM
        self.num_mixtures = num_mixtures
        rnn_params_dict[self.rnn_type]["num_mixtures"] = num_mixtures

        # Initialize the parameters of the RNN
        self.rnn = RNN_model(**rnn_params_dict[self.rnn_type]).to(self.device)

        # Initialize various betas, means and variances of the estimator

        # Posterior parameters
        self.beta_xt_yt_current = None # GMM Update: added beta
        self.mu_xt_yt_current = None
        self.L_xt_yt_current = None

        # Marginal parameters
        self.beta_yt_current = None # GMM Update: added beta
        self.mu_yt_current = None
        self.L_yt_current = None

        # Prior parameters
        self.beta_xt_yt_prev = None # GMM Update: added beta
        self.mu_xt_yt_prev = None
        self.L_xt_yt_prev = None

        # Marginal parameters for all mixture components
        self.mu_yt_all_mix = None
        self.L_yt_all_mix = None
    
    def push_to_device(self, x):
        """ Push the given tensor to the device
        """
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def compute_prior_mean_vars(self, beta_xt_yt_prev, mu_xt_yt_prev, L_xt_yt_prev):
        self.beta_xt_yt_prev = beta_xt_yt_prev # GMM Update: Added beta
        self.mu_xt_yt_prev = mu_xt_yt_prev
        self.L_xt_yt_prev = create_diag(L_xt_yt_prev)
        return self.beta_xt_yt_prev, self.mu_xt_yt_prev, self.L_xt_yt_prev

    def compute_marginal_mean_vars(self, beta_xt_yt_prev, mu_xt_yt_prev, L_xt_yt_prev, Cwi_batch):
        
        self.beta_yt_current = beta_xt_yt_prev
        self.mu_yt_all_mix = torch.einsum('ij,mntj->mnti',self.H, mu_xt_yt_prev)
        self.mu_yt_current = torch.einsum('mnti,mntj->ntj',beta_xt_yt_prev, self.mu_yt_all_mix) + self.mu_w

        self.L_yt_all_mix = self.H @ L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + Cwi_batch.unsqueeze(1).unsqueeze(0) # 5 dims
        L_wsum_current = torch.einsum('mnti,mntjk->ntjk',beta_xt_yt_prev, self.L_yt_all_mix)
        # Compute deviation part of the covariance
        mu_diff = self.mu_yt_all_mix - self.mu_yt_current.unsqueeze(0) # Unsqueeze adds 0:th dimension. Output shape: (2, 64, 60, 16)
        # Meeting Note: matmul instead of * (Maybe not, works)
        outerprod_dev = mu_diff.unsqueeze(-1) * mu_diff.unsqueeze(-2) # Unsqueeze adds last dimension and second to last (2, 64, 60, 16, 1), (2, 64, 60, 1, 16)
                                                                      # Represents transpose mult. Output shape: (2, 64, 60, 16, 16)
        # Using einsum to compute the weighted sum of outer products
        dev_wsum_current = torch.einsum('mnti,mntjk->ntjk', beta_xt_yt_prev, outerprod_dev)
        
        # Combine both parts to get the final covariance
        self.L_yt_current = L_wsum_current + dev_wsum_current # OP dims: (64,60,16,16) 
        
    # Meeting Note: Does not need beta as input, is stored in self like for mu and L    
    def compute_posterior_mean_vars(self, beta_xt_yt_prev, Yi_batch, Cwi_batch): # GMM Update: How does this change?
        Cwi_reshaped = Cwi_batch.unsqueeze(1).unsqueeze(0) # unsqueeze(1) accounts for T dim, and unsqueeze(0) for mix dim
        Re_t = self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + Cwi_reshaped # 5 dims
        Re_t_inv = torch.inverse(Re_t) # 5 dims
        self.K_t = self.L_xt_yt_prev @ (self.H.T @ Re_t_inv) # 5 dims
        self.eps_t = Yi_batch.unsqueeze(0) - torch.einsum('ij,mntj->mnti',self.H,self.mu_xt_yt_prev) - self.mu_w # 4 dims

        p = Yi_batch.shape[-1]  # dimensionality of the observations
        sqrt_2pi_p = torch.sqrt(torch.tensor((2 * math.pi) ** p))

        # log beta calc
        log_beta_post_num = torch.log(beta_xt_yt_prev.squeeze(-1)) - torch.log(sqrt_2pi_p * torch.sqrt(torch.det(Re_t))) \
            - 0.5 * torch.einsum('mnti, mnti->mnt', self.eps_t, torch.einsum('mntij,mntj->mnti', Re_t_inv, self.eps_t))
        log_beta_post = log_beta_post_num - torch.logsumexp(log_beta_post_num, dim=0, keepdim=True)
        
        self.beta_xt_yt_current = torch.exp(log_beta_post)

        mu_xt_yt_cur_all_mix = self.mu_xt_yt_prev + torch.einsum('mntij,mntj->mnti',self.K_t,self.eps_t) # 4 dims
        self.mu_xt_yt_current = torch.einsum('mnt,mntj->ntj',self.beta_xt_yt_current, mu_xt_yt_cur_all_mix) # 3 dims

        L_xt_yt_cur_all_mix = self.L_xt_yt_prev - (torch.einsum('mntij,mntjk->mntik',
                            self.K_t, Re_t) @ torch.transpose(self.K_t, 3, 4)) # 5 dims
      
        L_xt_yt_cur_wsum = torch.einsum('mnt,mntjk->ntjk', self.beta_xt_yt_current, L_xt_yt_cur_all_mix)
        # Mean deviation
        mu_diff = mu_xt_yt_cur_all_mix - self.mu_xt_yt_current.unsqueeze(0)
        outerprod_dev = mu_diff.unsqueeze(-1) * mu_diff.unsqueeze(-2)
        dev_xt_yt_cur_wsum = torch.einsum('mnt,mntjk->ntjk', self.beta_xt_yt_current, outerprod_dev)
        self.L_xt_yt_current = L_xt_yt_cur_wsum + dev_xt_yt_cur_wsum
        
        
        
        return self.beta_xt_yt_current, self.mu_xt_yt_current, self.L_xt_yt_current
    
    def compute_logpdf_Gaussian(self, Y, Cw): 
        _, T, _ = Y.shape
        Cw_reshaped = Cw.unsqueeze(1).unsqueeze(0)
        logprob = 0.5 * self.n_obs * math.log(math.pi*2) - 0.5 * torch.logdet(self.L_yt_all_mix+Cw_reshaped) \
            - 0.5 * torch.einsum('mnti,mnti->mnt',
            (Y.unsqueeze(0) - self.mu_yt_all_mix), 
            torch.einsum('mntij,mntj->mnti',torch.inverse(self.L_yt_all_mix+Cw_reshaped), 
                         (Y.unsqueeze(0) - self.mu_yt_all_mix)))

        return logprob
    
    # beta has shape (2,64,60,1) and logprob has shape m,n,t
    def compute_gmm_logpdf(self, logprob): 
        gmm_logprob = torch.logsumexp(torch.log(self.beta_xt_yt_prev.squeeze(-1) + 1e-8) + logprob, dim=0)
        return gmm_logprob.mean(1).mean(0) # shape scalar

    def compute_predictions(self, Y_test_batch, Cw_test_batch): 

        beta_x_given_Y_test_batch, mu_x_given_Y_test_batch, vars_x_given_Y_test_batch = self.rnn.forward(x=Y_test_batch)
        beta_xt_yt_prev_test, mu_xt_yt_prev_test, L_xt_yt_prev_test = self.compute_prior_mean_vars(
            beta_xt_yt_prev=beta_x_given_Y_test_batch,
            mu_xt_yt_prev=mu_x_given_Y_test_batch,
            L_xt_yt_prev=vars_x_given_Y_test_batch
            )
        beta_xt_yt_current_test, mu_xt_yt_current_test, L_xt_yt_current_test = self.compute_posterior_mean_vars(
                                                                                       beta_xt_yt_prev=beta_xt_yt_prev_test,
                                                                                       Yi_batch=Y_test_batch,
                                                                                       Cwi_batch=Cw_test_batch)
        return beta_x_given_Y_test_batch, mu_xt_yt_prev_test, L_xt_yt_prev_test,\
              beta_xt_yt_current_test, mu_xt_yt_current_test, L_xt_yt_current_test

    def forward(self, Yi_batch, Cwi_batch):

        beta_batch, mu_batch, vars_batch = self.rnn.forward(x=Yi_batch)
        beta_xt_yt_prev, mu_xt_yt_prev, L_xt_yt_prev = self.compute_prior_mean_vars(beta_xt_yt_prev=beta_batch, 
                                                                                    mu_xt_yt_prev=mu_batch,
                                                                                    L_xt_yt_prev=vars_batch)
        self.compute_marginal_mean_vars(beta_xt_yt_prev=beta_xt_yt_prev,
                                        mu_xt_yt_prev=mu_xt_yt_prev,
                                        L_xt_yt_prev=L_xt_yt_prev,
                                        Cwi_batch=Cwi_batch)
        
        # GMM Update: We need to return the gmm_logprob somewhere
        logprob_batch = self.compute_logpdf_Gaussian(Y=Yi_batch, Cw=Cwi_batch) # Per dim. and per sequence length
        logprob_gmm_batch = self.compute_gmm_logpdf(logprob_batch)

        return logprob_gmm_batch

# Rotnist Update: Change for rotnist
def train_danse(model, options, train_loader, val_loader, nepochs, logfile_path, modelfile_path, save_chkpoints, device='cpu', tr_verbose=False):
    
    # Push the model to device and count parameters
    model = push_model(nets=model, device=device)
    total_num_params, total_num_trainable_params = count_params(model)
    
    # Set the model to training
    model.train()
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model.rnn.lr)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs//3, gamma=0.9) # gamma was initially 0.9
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs//6, gamma=0.9) # gamma is now set to 0.9 OG: 6
    tr_losses = []
    val_losses = []
    val_mse_losses = []

    if modelfile_path is None:
        model_filepath = "./models/"
    else:
        model_filepath = modelfile_path

    #if save_chkpoints == True:
    if save_chkpoints == "all" or save_chkpoints == "some":
        # No grid search
        if logfile_path is None:
            training_logfile = "./log/danse_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path

    elif save_chkpoints == None:
        # Grid search
        if logfile_path is None:
            training_logfile = "./log/gs_training_danse_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path
    
    # Call back parameters
    
    patience = 0
    num_patience = 3 
    min_delta = options['rnn_params_dict'][model.rnn_type]["min_delta"] # 1e-3 for simpler model, for complicated model we use 1e-2
    #min_tol = 1e-3 # for tougher model, we use 1e-2, easier models we use 1e-5
    check_patience=False
    best_val_loss = np.inf
    tr_loss_for_best_val_loss = np.inf
    best_model_wts = None
    best_val_epoch = None
    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp
    

    # Convergence monitoring (checks the convergence but not ES of the val_loss)
    model_monitor = ConvergenceMonitor(tol=min_delta,
                                    max_epochs=num_patience)

    # This checkes the ES of the val loss, if the loss deteriorates for specified no. of
    # max_epochs, stop the training
    #model_monitor = ConvergenceMonitor_ES(tol=min_tol, max_epochs=num_patience)

    print("------------------------------ Training begins --------------------------------- \n")
    print("Config: {} \n".format(options))
    print("\n Config: {} \n".format(options), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))

    # Start time
    starttime = timer()
    try:
        for epoch in range(nepochs):
            
            tr_running_loss = 0.0
            tr_loss_epoch_sum = 0.0
            val_loss_epoch_sum = 0.0
            val_mse_loss_epoch_sum = 0.0
        
            for i, data in enumerate(train_loader, 0):
            
                tr_Y_batch, tr_Z_batch, tr_Cw_batch = data
                optimizer.zero_grad()
                Y_train_batch = Variable(tr_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                Cw_train_batch = Variable(tr_Cw_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                log_pY_train_batch = -model.forward(Y_train_batch, Cw_train_batch)
                log_pY_train_batch.backward()
                optimizer.step()

                # print statistics
                tr_running_loss += log_pY_train_batch.item()
                tr_loss_epoch_sum += log_pY_train_batch.item()

                if i % 100 == 99 and ((epoch + 1) % 100 == 0):    # print every 10 mini-batches
                    #print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 100))
                    #print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 100), file=orig_stdout)
                    tr_running_loss = 0.0
            
            scheduler.step()

            endtime = timer()
            # Measure wallclock time
            time_elapsed = endtime - starttime

            with torch.no_grad():
                
                for i, data in enumerate(val_loader, 0):
                    
                    val_Y_batch, val_Z_batch, val_Cw_batch = data
                    Y_val_batch = Variable(val_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                    Cw_val_batch = Variable(val_Cw_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                    val_beta_Z_predictions_batch, val_mu_Z_predictions_batch, val_var_Z_predictions_batch, \
                    val_beta_Z_filtered_batch, val_mu_Z_filtered_batch, val_var_Z_filtered_batch = model.compute_predictions(Y_val_batch, Cw_val_batch)
                    log_pY_val_batch = -model.forward(Y_val_batch, Cw_val_batch)
                    val_loss_epoch_sum += log_pY_val_batch.item()
                    #val_mse_loss_batch = mse_criterion(val_X_batch[:,1:,:].to(device), val_mu_X_filtered_batch)
                    val_mse_loss_batch = mse_criterion(val_Z_batch.to(device), val_mu_Z_filtered_batch)
                    # print statistics
                    val_mse_loss_epoch_sum += val_mse_loss_batch.item()


            # Loss at the end of each epoch
            tr_loss = tr_loss_epoch_sum / len(train_loader)
            val_loss = val_loss_epoch_sum / len(val_loader)
            val_mse_loss = val_mse_loss_epoch_sum / len(val_loader)

            # Record the validation loss per epoch
            if (epoch + 1) > nepochs // 3: # nepochs/6 for complicated, 100 for simpler model
                model_monitor.record(val_loss)

            # Displaying loss at an interval of 200 epochs
            if tr_verbose == True and (((epoch + 1) % 50) == 0 or epoch == 0):
                
                print("Epoch: {}/{}, Training NLL:{:.9f}, Val. NLL:{:.9f}, Val. MSE:{:.9f}".format(epoch+1, 
                model.rnn.num_epochs, tr_loss, val_loss, val_mse_loss), file=orig_stdout)
                #save_model(model, model_filepath + "/" + "{}_ckpt_epoch_{}.pt".format(model.model_type, epoch+1))

                print("Epoch: {}/{}, Training NLL:{:.9f}, Val. NLL:{:.9f}, Val. MSE: {:.9f}, Time_Elapsed:{:.4f} secs".format(epoch+1, 
                model.rnn.num_epochs, tr_loss, val_loss, val_mse_loss, time_elapsed))
            
            # Checkpointing the model every few  epochs
            #if (((epoch + 1) % 500) == 0 or epoch == 0) and save_chkpoints == True:     
            if (((epoch + 1) % 100) == 0 or epoch == 0) and save_chkpoints == "all": 
                # Checkpointing model every few epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(model, model_filepath + "/" + "danse_{}_ckpt_epoch_{}.pt".format(model.rnn_type, epoch+1))
            elif (((epoch + 1) % nepochs) == 0) and save_chkpoints == "some": 
                # Checkpointing model at the end of training epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(model, model_filepath + "/" + "danse_{}_ckpt_epoch_{}.pt".format(model.rnn_type, epoch+1))
            
            # Save best model in case validation loss improves
            '''
            best_val_loss, best_model_wts, best_val_epoch, patience, check_patience = callback_val_loss(model=model,
                                                                                                    best_model_wts=best_model_wts,
                                                                                                    val_loss=val_loss,
                                                                                                    best_val_loss=best_val_loss,
                                                                                                    best_val_epoch=best_val_epoch,
                                                                                                    current_epoch=epoch+1,
                                                                                                    patience=patience,
                                                                                                    num_patience=num_patience,
                                                                                                    min_delta=min_delta,
                                                                                                    check_patience=check_patience,
                                                                                                    orig_stdout=orig_stdout)
            if check_patience == True:
                print("Monitoring validation loss for criterion", file=orig_stdout)
                print("Monitoring validation loss for criterion")
            else:
                pass
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss # Save best validation loss
                tr_loss_for_best_val_loss = tr_loss # Training loss corresponding to best validation loss
                best_val_epoch = epoch+1 # Corresponding value of epoch
                best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model
            '''
            # Saving every value
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            val_mse_losses.append(val_mse_loss)

            
            # Check monitor flag
            if model_monitor.monitor(epoch=epoch+1) == True:

                if tr_verbose == True:
                    print("Training convergence attained! Saving model at Epoch: {}".format(epoch+1), file=orig_stdout)
                
                print("Training convergence attained at Epoch: {}!".format(epoch+1))
                # Save the best model as per validation loss at the end
                best_val_loss = val_loss # Save best validation loss
                tr_loss_for_best_val_loss = tr_loss # Training loss corresponding to best validation loss
                best_val_epoch = epoch+1 # Corresponding value of epoch
                best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model
                #print("\nSaving the best model at epoch={}, with training loss={}, validation loss={}".format(best_val_epoch, tr_loss_for_best_val_loss, best_val_loss))
                #save_model(model, model_filepath + "/" + "{}_usenorm_{}_ckpt_epoch_{}.pt".format(model.model_type, usenorm_flag, epoch+1))
                break

            #else:

                #print("Model improvement attained at Epoch: {}".format(epoch+1))
                #best_val_loss = val_loss # Save best validation loss
                #tr_loss_for_best_val_loss = tr_loss # Training loss corresponding to best validation loss
                #best_val_epoch = epoch+1 # Corresponding value of epoch
                #best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model

            
        # Save the best model as per validation loss at the end
        print("\nSaving the best model at epoch={}, with training loss={}, validation loss={}".format(best_val_epoch, tr_loss_for_best_val_loss, best_val_loss))
        
        #if save_chkpoints == True:
        if save_chkpoints == "all" or save_chkpoints == "some":
            # Save the best model using the designated filename
            if not best_model_wts is None:
                model_filename = "danse_{}_ckpt_epoch_{}_best.pt".format(model.rnn_type, best_val_epoch)
                torch.save(best_model_wts, model_filepath + "/" + model_filename)
            else:
                model_filename = "danse_{}_ckpt_epoch_{}_best.pt".format(model.rnn_type, epoch+1)
                print("Saving last model as best...")
                save_model(model, model_filepath + "/" + model_filename)
        #elif save_chkpoints == False:
        elif save_chkpoints == None:
            pass
    
    except KeyboardInterrupt:

        if tr_verbose == True:
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1), file=orig_stdout)
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1))
        else:
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1))
        
        if not save_chkpoints is None:
            model_filename = "danse_{}_ckpt_epoch_{}_latest.pt".format(model.rnn_type, epoch+1)
            torch.save(model, model_filepath + "/" + model_filename)

    print("------------------------------ Training ends --------------------------------- \n")
    # Restoring the original std out pointer
    sys.stdout = orig_stdout

    return tr_losses, val_losses, val_mse_losses, best_val_loss, tr_loss_for_best_val_loss, model

