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
- [ ] Plot X sequence, z sequence and y sequence for DANSE in test_danse_rotnist file
- [x] Implement PSNR and SSIM comparison plots
- [ ] Modify `parameters_opt.py` and tweak rnn params