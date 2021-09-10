# Change it to upsampling conv2d

import os
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

def fn(yhat, y):
    return torch.norm(yhat.reshape(-1,) - y) ** 2 
    # return torch.norm(yhat.reshape(-1) - y) ** 2 / 2

def run(num_iter=100000, num_trials=100000,
        Nt=8, Nr=8, M=16, GD_lr=0.001, SNR_dB=10, 
        results_dir=None, datasets_dir=None, savedataset=False, writer=None, ser=None,
        seed=42):
    
    # Generate X, Y, H using random function or from dataset
    (X, Y, H, _) = generate(num_trials, Nt, Nr, M, SNR_dB, seed, rootdir=datasets_dir, save=savedataset)
    
    # Build autoencoder architecture neural network --> Coba liat punya Jeffrey Andrews
    G = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(4),
        nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1), # --> this is used in results0
        # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), # --> this is used in results0
        # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(8), # --> this is used in results0
        # nn.BatchNorm2d(16),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1), # --> this is used in results0
        # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(4), # --> this is used in results0
        # nn.BatchNorm2d(16),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1), # --> this is used in results0
        # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(2), # --> this is used in results0
        # nn.BatchNorm2d(16),
        nn.Flatten(),
        nn.Linear(64**2*2, Nt*2),
        ).type(dtype)
    
    opt = optim.Adam(G.parameters(), lr=GD_lr)
    
    list_mse_xhat_x = []
    list_fidelity_loss = []
    list_ser = []
    lines = f'num_iter,fidelity_loss,mse_xhat_x,SER\n'
    for trial in range(num_trials):
        b = torch.from_numpy(Y[trial]).type(dtype).reshape(-1,)
        
        Htorch = torch.from_numpy(H[trial]).reshape(1,1,2*Nr,2*Nt).type(dtype)
        
        G.apply(weight_reset)
        z = torch.zeros_like(torch.empty(1,2,64,64)).type(dtype).normal_() # number of channel (dim=1) must be same with number of channel input in the neural network
        z.requires_grad = False
        
        # Record dictionary
        record = {"mse_xhat_x": [],
                  "fidelity_loss": [],
                  "SER": []
                  }

        for t in range(num_iter):
            # Run DIP
            X_hat = G(z).reshape(1,1,-1,1) # *2-1
            # X_hat = torch.transpose(X_hat,1,2).reshape(1,1,-1,1)*2-1
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
        # writer = SaveRecord(record, writer, ts, t=t)
        print('record %0.2f trial %2d value: %e' % (SNR_dB,trial+1,record["mse_xhat_x"][t]))
   
        # mat_xhat = np.append(mat_xhat, X_hat_numpy).reshape(-1,1)
        # mat_xtrue = np.append(mat_xtrue, X[trial]).reshape(-1,1)
        
    arr_mse_xhat_x = np.array(list_mse_xhat_x)
    arr_fidelity_loss = np.array(list_fidelity_loss)
    arr_ser = np.array(list_ser)
    for t in range(num_iter):
        if (t+1) % 100 == 0:
            lines = lines + f'{t+1},{np.mean(arr_fidelity_loss[:,t]):.8f},{np.mean(arr_mse_xhat_x[:,t]):.8f},{np.mean(arr_ser[:,t]):.8f}\n'
    fout = open(results_dir+"ser_%0.2f_dB.txt" % SNR_dB, 'w')
    fout.write(lines)
    fout.close()
    # writer.save()


# Main program
tic = time.time()

# Check if cuda is available or not
if torch.cuda.is_available():
    dtype = torch.cuda.DoubleTensor
else:
    dtype = torch.DoubleTensor

datasets_dir = '../data/AutoCNNLinearRelUBatchNorm/'
if not os.path.isdir(datasets_dir):
    os.makedirs(datasets_dir)
results_dir = '../data/AutoCNNLinearRelUBatchNorm/results/'

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
