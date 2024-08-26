# DANSE GMM Readme
This README is a temporary one to keep abreast of the changes and progress maded while implementing DANSE GMM.

Things to do before Friday 19/7
- [x]	Consider if we should change to pkl from pt
- [x]	Fix DANSE for rotnist by changing:
    - [x]	Danse_rotnist.py (check if the data is being loaded in sequenceor is it shuffled)
    - [x]	Main_danse_opt_rotnist.py
    - [x]	Utils_rotnist.py
    - [x]	Parameters_opt.py (tweak for rotnist)
- [x] Decoding to show y for different smnr
- [x] Look into the C_w matrix size

Things to do before Friday 26/7
- [x] Plot log files loss trend by taking the json outputs
- [x] Change the recreate_latent_values in test_danse_rotnist.py to use posterior mean for z
- [x] Modify generation of X images to account for test data as well
- [x] Add X sequence to the  z sequence - y sequence plot for DANSE
    - [x] Decode Z_LS to obtain X_LS for comparison plots
- [x] Implement PSNR and SSIM comparison plots
- [ ] Modify `parameters_opt.py` and tweak rnn params
- Look at the time schedule and what to do

Meeting 26/7 notes
- [x] Keep mse nmse for z, zhat also add for x, xhat
- [x] Change plots to show x, xhat_ls, xhat_danse, show 10 and separate plots for each noise dB
- [x] Tune parameters_opt 
- [x] Try noise for -5, 0, 5, 10 dB

Meeting 29/7
- [x] Trying longer t lengths, (t=60) with lower latent dim
- [x] Tune parameters for t=60, latent_dim=16
- [x] Start working on GMM, create outline

GMM Work
- [x] Modify train_test_split to account only for validation and train sequences (test not reqd)
- [x] Modify `rnn_gmm.py` file to output 4dim triplets of (beta, mean, cov)
- [x] Modify `danse_gmm.py` file with updated forward, prior, posterior and comp_pred funcs
- [x] Add danse_gmm option to `parameters_opt.py` file and modify as required
- [x] Maintain `main_danse_gmm_rotnist.py` to be in sync with danse_gmm changes

Meeting 12/8
- [x] Add the Cw before (see note in the file)
- [x] Change beta calculations to log
- [x] Move beta before mean and cov
- [x] Add danse_rotnist to test

Meeting 23/8
- [ ] return posterior betas into log file
- [ ] change naming structure for figs to include n_mix

Future:
- [ ] Test for a more controlled experiment with multivariate Gaussians as the underlying dist

Things to try if nothing else works
- Really short sequences (is it getting better with the longer sequences?)
- Train on just one kind of number
- Calculating the mse loss for the X instead by decoding within the training loop (maybe not)