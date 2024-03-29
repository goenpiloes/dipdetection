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
import time
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class Autoencoder(nn.Module):
    def __init__(self,Nt):
        # N, cotto
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(Nt*2,16),
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
            nn.Linear(16,Nt*2),
            nn.Sigmoid()
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def fn(yhat, y):
    return torch.norm(yhat.reshape(-1,) - y) ** 2 
    # return torch.norm(yhat.reshape(-1) - y) ** 2 / 2

def run(num_iter=100000, num_trials=100000,
        Nt=8, Nr=8, M=16, GD_lr=0.001, SNR_dB=10, 
        results_dir=None, datasets_dir=None, savedataset=False, writer=None, ser=None,
        seed=42):

    # Generate X, Y, H using random function or from dataset
    (X, Y, H, _) = generate(num_trials, Nt, Nr, M, SNR_dB, seed, rootdir=datasets_dir, save=savedataset)
     
    # G = skip(1, 1, 
    #          num_channels_down=[16, 32, 64, 128, 128],
    #          num_channels_up=[16, 32, 64, 128, 128], # [16, 32, 64, 128, 128],
    #          num_channels_skip=[0, 0, 0, 0, 0],
    #          filter_size_up=3, filter_size_down=3, filter_skip_size=1,
    #          upsample_mode='nearest',  # downsample_mode='avg',
    #          need1x1_up=False,
    #          need_sigmoid=True, need_bias=True, pad='zero', act_fun='LeakyReLU')
    
    G = Autoencoder(Nt).type(dtype)
    
    opt = optim.Adam(G.parameters(), lr=GD_lr)
    
    # mat_xtrue = np.array([])
    # mat_xhat = np.array([])
    list_mse_xhat_x = []
    list_fidelity_loss = []
    list_ser = []
    lines = f'num_iter,fidelity_loss,mse_xhat_x,SER\n'
    for trial in range(num_trials):
        b = torch.from_numpy(Y[trial]).type(dtype).reshape(-1,)
        Htorch = torch.from_numpy(H[trial]).reshape(1,1,2*Nr,2*Nt).type(dtype)
        
        G.apply(weight_reset)
        z = torch.zeros_like(torch.empty(1,Nt*2)).type(dtype).normal_()
        z.requires_grad = False
        
        # Record dictionary
        record = {"mse_xhat_x": [],
                  "fidelity_loss": [],
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
            
            mse_gt = np.mean((X_hat_numpy - X[trial]) ** 2)
            ser_instant = CalcSER(X[trial], X_hat_numpy, M)
            
            # Record the result
            record["mse_xhat_x"].append(mse_gt)
            record["fidelity_loss"].append(fidelity_loss.item())
            record["SER"].append(ser_instant)
        
        list_mse_xhat_x.append(record['mse_xhat_x'])
        list_fidelity_loss.append(record['fidelity_loss'])
        list_ser.append(record['SER'])
    
        # # Save the process
        # writer = SaveRecord(record, writer, trial, t=t)
        print('record %0.2f trial %2d value: %e' % (SNR_dB,trial+1,record["mse_xhat_x"][t]))

    arr_mse_xhat_x = np.array(list_mse_xhat_x)
    arr_fidelity_loss = np.array(list_fidelity_loss)
    arr_ser = np.array(list_ser)
    for t in range(num_iter):
        if (t+1) % 100 == 0:
            lines = lines + f'{t+1},{np.mean(arr_fidelity_loss[:,t]):.8f},{np.mean(arr_mse_xhat_x[:,t]):.8f},{np.mean(arr_ser[:,t]):.8f}\n'
    # ser["%d_dB" % SNR_dB] = CalcSER(mat_xtrue, mat_xhat, M)
    fout = open(results_dir+"ser_%0.2f_dB.txt" % SNR_dB, 'w')
    fout.write(lines)
    fout.close()


# Main program
tic = time.time()

# Check if cuda is available or not
if torch.cuda.is_available():
    dtype = torch.cuda.DoubleTensor
else:
    dtype = torch.DoubleTensor

datasets_dir = '../data/AutoLinearLeakyReLU/'
if not os.path.isdir(datasets_dir):
    os.makedirs(datasets_dir)
results_dir = '../data/AutoLinearLeakyReLU/results/'

# Create SNR_dB array
# arrSNR_dB = np.array([i for i in range(0,36,5)])
arrSNR_dB = np.array([30])
ser = dict()
ser1 = dict()
for SNR_dB in arrSNR_dB:
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # writerdata = pd.ExcelWriter(results_dir + 'results_%0.2f_dB.xlsx' % SNR_dB, engine='xlsxwriter')
    run(num_iter=2000,
        num_trials=100,
        Nt=8, Nr=8, M=4,
        GD_lr=0.001,
        SNR_dB=SNR_dB, 
        results_dir = results_dir,
        datasets_dir = datasets_dir,
        savedataset=False, writer=None, ser = ser,
        seed=None)
toc = time.time()
print(f'Time spent by the system: {toc-tic}')
