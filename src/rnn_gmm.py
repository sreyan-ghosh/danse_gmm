import numpy as np
import scipy
import sys
import torch
from torch import nn
import torch.nn.functional as F

# Create an RNN model for prediction
class RNN_model(nn.Module):
    """ This super class defines the specific model to be used i.e. LSTM or GRU or RNN
    """
    def __init__(self, input_size, output_size, num_mixtures, n_hidden, n_layers, 
        model_type, lr, num_epochs, n_hidden_dense=32, num_directions=1, batch_first=True, min_delta=1e-2, device='cpu'):
        super(RNN_model, self).__init__()
        """
        Args:
        - input_size: The dimensionality of the input data
        - output_size: The dimensionality of the output data
        - n_hidden: The size of the hidden layer, i.e. the number of hidden units used
        - n_layers: The number of hidden layers
        - model_type: The type of model used ("lstm"/"gru"/"rnn")
        - lr: Learning rate used for training
        - num_epochs: The number of epochs used for training
        - num_directions: Parameter for bi-directional RNNs (usually set to 1 in this case, for bidirectional set as 2)
        - batch_first: Option to have batches with the batch dimension as the starting dimension 
        of the input data
        """
        # Defining some parameters
        self.hidden_dim = n_hidden  
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.model_type = model_type
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        
        # Predefined:
        # Use only the forward direction 
        self.num_directions = num_directions
        
        # The input tensors must have shape (batch_size,...)
        self.batch_first = batch_first
        
        # Defining the recurrent layers 
        if model_type.lower() == "rnn":
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)   
        elif model_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)
        elif model_type.lower() == "gru":
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)  
        else:
            print("Model type unknown:", model_type.lower()) 
            sys.exit() 
        
        # Fully connected layer to be used for mapping the output
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, n_hidden_dense).to(self.device)
        self.fc_mean = nn.Linear(n_hidden_dense, self.output_size * self.num_mixtures).to(self.device)
        self.fc_vars = nn.Linear(n_hidden_dense, self.output_size * self.num_mixtures).to(self.device)
        self.fc_weights = nn.Linear(n_hidden_dense, self.num_mixtures).to(self.device)

    def init_h0(self, batch_size):
        """ This function defines the initial hidden state of the RNN
        """
        # This method generates the first hidden state of zeros (h0) which is used in the forward pass
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return h0
    
    def forward(self, x):
        """ This function defines the forward function to be used for the RNN model
        """
        batch_size = x.shape[0]
        
        # Obtain the RNN output
        r_out, _ = self.rnn(x)
        
        # Reshaping the output appropriately
        r_out_all_steps = r_out.contiguous().view(batch_size, -1, self.num_directions * self.hidden_dim)

        # Passing the output to one fully connected layer
        y = F.relu(self.fc(r_out_all_steps))

        # Means and variances are computed for time instants t=2, ..., T+1 using the available sequence
        mu_2T_1 = self.fc_mean(y).view(batch_size, -1, self.num_mixtures, self.output_size)  # (batch_size, seq_len, num_mixtures, output_size)
        vars_2T_1 = F.softplus(self.fc_vars(y)).view(batch_size, -1, self.num_mixtures, self.output_size)  # (batch_size, seq_len, num_mixtures, output_size)
        beta_2T_1 = F.softmax(self.fc_weights(y), dim=-1).view(batch_size, -1, self.num_mixtures)  # (batch_size, seq_len, num_mixtures)

        # The mean, variances, and weights at the first time step need to be computed only based on the previous hidden state
        mu_1 = self.fc_mean(F.relu(self.fc(self.init_h0(batch_size)[-1,:,:]))).view(batch_size, 1, self.num_mixtures, self.output_size)
        var_1 = F.softplus(self.fc_vars(F.relu(self.fc(self.init_h0(batch_size)[-1,:,:])))).view(batch_size, 1, self.num_mixtures, self.output_size)
        beta_1 = F.softmax(self.fc_weights(F.relu(self.fc(self.init_h0(batch_size)[-1,:,:]))), dim=-1).view(batch_size, 1, self.num_mixtures)

        # To get the means, variances, and weights for the time instants t=1, ..., T, we take the previous result and concatenate 
        # all but last value to the value found at t=1. Concatenation is done along the sequence dimension
        mu = torch.cat(
            (mu_1, mu_2T_1[:,:-1,:,:]),
            dim=1
        )

        vars = torch.cat(
            (var_1, vars_2T_1[:,:-1,:,:]),
            dim=1
        )

        beta = torch.cat(
            (beta_1, beta_2T_1[:,:-1,:]),
            dim=1
        )

        # Permuting the dimensions to match the desired output format
        mu = mu.permute(2, 0, 1, 3)  # (num_mixtures, batch_size, sequence_length, output_size)
        vars = vars.permute(2, 0, 1, 3)  # (num_mixtures, batch_size, sequence_length, output_size)
        beta = beta.permute(2, 0, 1).unsqueeze(-1)  # (num_mixtures, batch_size, sequence_length, 1)

        return beta, mu, vars # Right now, dims = (nmix, batchsz, seqlen, opsz)