# Change it to autoencoder MLP

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append('../')
from tools import *
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt
import time
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class Autoencoder(nn.Module):
    def __init__(self):
        # N, cotto
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64,16),
            nn.LeakyReLU(),
            nn.Linear(16,8),
            nn.LeakyReLU(),
            nn.Linear(8,8),
            nn.LeakyReLU(),
            nn.Linear(8,2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2,8),
            nn.LeakyReLU(),
            nn.Linear(8,8),
            nn.LeakyReLU(),
            nn.Linear(8,16),
            nn.LeakyReLU(),
            nn.Linear(16,64),
            nn.Sigmoid()
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def fn(yhat, y):
    return torch.norm(yhat.reshape(-1,) - y) ** 2 
    # return torch.norm(yhat.reshape(-1) - y) ** 2 / 2

def run(num_iter=100000, num_samples=100000,
        Nt=8, Nr=8, M=16, GD_lr=0.001, SNR_dB=10, 
        results_dir=None, datasets_dir=None, savedataset=False, writer=None, ser=None,
        seed=42):
    
    # Generate X, Y, H using random function or from dataset
    (X, Y, H, _) = generate(num_samples, Nt, Nr, M, SNR_dB, seed, rootdir=datasets_dir, save=savedataset)
    
    # X = X.reshape(-1,1,2*Nt,1)
    # Y = Y.reshape(-1,1,2*Nr,1)
    # H = H.reshape(-1,1,2*Nr,2*Nt)
    
    # G = skip(1, 1, 
    #          num_channels_down=[16, 32, 64, 128, 128],
    #          num_channels_up=[16, 32, 64, 128, 128], # [16, 32, 64, 128, 128],
    #          num_channels_skip=[0, 0, 0, 0, 0],
    #          filter_size_up=3, filter_size_down=3, filter_skip_size=1,
    #          upsample_mode='nearest',  # downsample_mode='avg',
    #          need1x1_up=False,
    #          need_sigmoid=True, need_bias=True, pad='zero', act_fun='LeakyReLU')
    
    G = Autoencoder().type(dtype)
    
    opt = optim.Adam(G.parameters(), lr=GD_lr)
    
    mat_xtrue = np.array([])
    mat_xhat = np.array([])
    for ts in range(num_samples):
        b = torch.from_numpy(Y[ts]).type(dtype).reshape(-1,)
        
        Htorch = torch.from_numpy(H[ts]).reshape(1,1,2*Nr,2*Nt).type(dtype)
        
        G.apply(weight_reset)
        z = torch.zeros_like(torch.empty(1,Nt*2)).type(dtype).normal_()
        z.requires_grad = False
        
        # Record dictionary
        record = {"mse_xhat_x": [],
                  "fidelity_loss": [],
                  "cpu_time": [],
                  "SER": []
                  }

        for t in range(num_iter):
            # Run DIP
            X_hat = G(z)
            X_hat = X_hat.reshape(1,1,-1,1)*2-1
            Y_hat = torch.matmul(Htorch,X_hat)
            fidelity_loss = fn(Y_hat,b)

            opt.zero_grad()
            fidelity_loss.backward()
            opt.step()

            X_hat_numpy = X_hat.detach().cpu().numpy().reshape((-1,1))
            
            mse_gt = np.mean((X_hat_numpy - X[ts]) ** 2)
            ser_instant = CalcSER(X[ts], X_hat_numpy, M)
            
            # Record the result
            record["mse_xhat_x"].append(mse_gt)
            record["fidelity_loss"].append(fidelity_loss.item())
            record["SER"].append(ser_instant)
            record["cpu_time"].append(time.time())
            # if (t + 1) % 100 == 0:
            #     print('trial %3d: Iter %5d  fidelity_loss: %e  MSE_xhat_x: %e' % (ts+1, t + 1, fidelity_loss.item(), mse_gt))
            
            # if (np.mean(np.abs(Y_hat_numpy-Y[ts])**2) < noisevar*1.05 and np.mean(np.abs(Y_hat_numpy-Y[ts])**2) > noisevar*0.95):
            #     result_stop = X_hat_numpy
            #     t_stop = t
            
        minvalue_gt = (record["mse_xhat_x"][t])
    
        # # Save the process
        writer = SaveRecord(record, writer, ts, t=t)
        print('record %0.2f trial %2d value: %e' % (SNR_dB,ts+1,minvalue_gt))
   
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

datasets_dir = '../data/test3/'
if not os.path.isdir(datasets_dir):
    os.makedirs(datasets_dir)
results_dir = '../data/test3/results_32x32/'

# Create SNR_dB array
# arrSNR_dB = np.array([i for i in range(0,36,5)])
arrSNR_dB = np.array([30])
ser = dict()
ser1 = dict()
for SNR_dB in arrSNR_dB:
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    writerdata = pd.ExcelWriter(results_dir + 'results_%0.2f_dB.xlsx' % SNR_dB, engine='xlsxwriter')
    run(num_iter=2000,
        num_samples=1000,
        Nt=32, Nr=32, M=16,
        GD_lr=0.001,
        SNR_dB=SNR_dB, 
        results_dir = results_dir,
        datasets_dir = datasets_dir,
        savedataset=False, writer=writerdata, ser = ser,
        seed=None)
toc = time.time()
print(f'Time spent by the system: {toc-tic}')
