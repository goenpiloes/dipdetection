'''
The first edition.
This won't be used later because it's slow to get the results.

Information you need to know:
    - The size of z never change during neural network process; we didn't do downsampling and also upsampling because the size is too small
    - the loss function (or fidelity loss) = torch.norm(b - H*X_hat)**2 where this norm is frobenius norm
'''


import os
from tools import *
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt
import time
from models import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# weight reset function
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# fidelity loss function
def fn(yhat, y):
    return torch.norm(yhat.reshape(-1) - y) ** 2 
    # return torch.norm(yhat.reshape(-1) - y) ** 2 / 2

# run function
def run(num_iter=100000, trial_size=100, 
        Nt=8, Nr=8, M=16, GD_lr=0.001, SNR_dB=10, 
        results_dir=None, datasets_dir=None, savedataset=False, writer=None, ser=None,
        seed=42):
    
    # Generate X, Y, H using random function or from dataset
    (X, Y, H, _) = generate(trial_size, Nt, Nr, M, SNR_dB, seed, rootdir=datasets_dir, save=savedataset)
    
    # This array will save all xhat and xtrue when doing simulation at one SNR value
    mat_xhat = np.array([])
    mat_xtrue = np.array([])
    
    G = skip(1, 1,
             num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],#[16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='zero', act_fun='LeakyReLU').type(dtype)
    
    for ts in range(trial_size):
        b = torch.from_numpy(Y[ts]).type(dtype).reshape(-1) # this is equal to y in tensor type
        
        Htorch = torch.from_numpy(H[ts]).reshape(1,1,2*Nr,2*Nt).type(dtype)
        
        G.apply(weight_reset) # reset weight parameter
        
        z = torch.zeros_like(torch.empty(1,1,Nr,2)).type(dtype).normal_() # create noise input
    
        z.requires_grad = False
        opt = optim.Adam(G.parameters(), lr=GD_lr)
        
        # Record dictionary
        record = {"mse_yhat_y": [],
                  "mse_xhat_x": [],
                  "fidelity_loss": [],
                   "cpu_time": [] # ,"SER": []
                  }
    
        for t in range(num_iter):
            # Run DIP algorithm
            X_hat = G(z).type(dtype)
            X_hat = (torch.transpose(X_hat,2,3).reshape(1,1,-1,1))*2-1 # we made it so the range of X_hat is [-1,1]
            Y_hat = torch.matmul(Htorch,X_hat)
            fidelity_loss = fn(Y_hat,b)
    
            opt.zero_grad()
            fidelity_loss.backward()
            opt.step()
    
            # Convert Y_hat and X_hat to numpy array type
            Y_hat_numpy = Y_hat.detach().cpu().numpy().reshape((-1,1), order='F')
            X_hat_numpy = X_hat.detach().cpu().numpy().reshape((-1,1), order='F')
            
            # calculate mse
            mse_yhat_y = np.mean((Y_hat_numpy - Y[ts]) ** 2)
            mse_gt = np.mean((X_hat_numpy - X[ts]) ** 2)
            
            # Record the result
            record["mse_yhat_y"].append(mse_yhat_y)
            record["mse_xhat_x"].append(mse_gt)
            record["fidelity_loss"].append(fidelity_loss.item())
            # record["SER"].append(ser)
            record["cpu_time"].append(time.time())
            if (t + 1) % 100 == 0:
                print('trial %3d: Iter %5d   MSE_yhat_y: %e MSE_xhat_x: %e' % (ts+1, t + 1, mse_yhat_y, mse_gt))
                # print('trial %3d: Iter %5d   MSE_yhat_y: %e MSE_gt: %e' % (ts+1, t + 1, mse_yhat_y, mse_gt))
                
        # Looking its best stopping point
        minvalue_nuh = (record["mse_yhat_y"][t])
        minvalue_gt = (record["mse_xhat_x"][t])
        
        # Save the process
        writer = SaveRecord(record, writer, ts, t)
        print('record %0.2f trial %2d value(%e, %e)' % (SNR_dB,ts+1,minvalue_nuh,minvalue_gt))
        
        mat_xhat = np.append(mat_xhat, X_hat_numpy).reshape(-1,1)
        mat_xtrue = np.append(mat_xtrue, X[ts]).reshape(-1,1)
        
    ser["%d_dB" % SNR_dB] = CalcSER(mat_xtrue, mat_xhat, M)
    firstline = f'The SER of SNR_dB = {SNR_dB} dB is {ser["%d_dB" % SNR_dB]}.'
    print(firstline)
    fout = open(results_dir+"ser_%0.2f_dB.txt" % SNR_dB, 'w')
    fout.write(firstline)
    fout.close()
    writer.save()
    

# Main program
tic = time.time()
# Check if cuda is available or not
if torch.cuda.is_available():
    dtype = torch.cuda.DoubleTensor
else:
    dtype = torch.DoubleTensor
    
datasets_dir = './data/main_old/'
if not os.path.isdir(datasets_dir):
    os.makedirs(datasets_dir)
results_dir = './data/main_old/results/'

# Create SNR_dB array
# arrSNR_dB = np.array([i for i in range(0,36,5)])
arrSNR_dB = np.array([25, 30])
ser = dict()
for SNR_dB in arrSNR_dB:
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    writerdata = pd.ExcelWriter(results_dir + 'results_%0.2f_dB.xlsx' % SNR_dB, engine='xlsxwriter')
    run(num_iter=1000,
        trial_size=100, 
        Nt=8, Nr=8, M=4,
        GD_lr=0.001, 
        SNR_dB=SNR_dB, 
        results_dir = results_dir,
        datasets_dir=datasets_dir,
        savedataset=True, writer=writerdata, ser = ser,
        seed=None)
toc = time.time()
print(f'Time spent by the system: {toc-tic}')
