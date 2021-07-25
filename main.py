import os
from tools import *
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
import time
from models import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def run(num_iter=100000, batch_size=100, 
        Nt=8, Nr=8, M=16, GD_lr=0.001, SNR_dB=10, 
        results_dir=None, datasets_dir=None, savedataset=False, 
        seed=42):
    # Generate X, Y, H using random function or from dataset
    (X, Y, H) = generate(batch_size, Nt, Nr, M, SNR_dB, seed, rootdir=datasets_dir, save=savedataset)
    # (X, Y, H) = pd.read_excel('users.xlsx', sheet_name = [0,1,2])

    for bs in range(batch_size):
        b_numpy = Y[bs].reshape((1,1,-1,2),order='F')
        b = torch.from_numpy(b_numpy).reshape(-1)
        
        Ht = H[bs].transpose(1,0)
        Hinv = np.dot(np.linalg.inv(np.dot(Ht, H[bs])), Ht)
        
        # Create model G and random noise input z
        G = skip(1, 1,
                 num_channels_down=[2, 4, 4, 8],
                 num_channels_up=[2, 4, 4, 8],#[16, 32, 64, 128, 128],
                 num_channels_skip=[0, 0, 0, 0],
                 filter_size_up=3, filter_size_down=3, filter_skip_size=1,
                 upsample_mode='nearest',  # downsample_mode='avg',
                 need1x1_up=False,
                 need_sigmoid=True, need_bias=True, pad='zero', act_fun='LeakyReLU').type(dtype)
        z = torch.zeros_like(torch.empty(1,1,Nr,2)).type(dtype).normal_()
    
        z.requires_grad = False
        opt = optim.Adam(G.parameters(), lr=GD_lr)
        
        # Record dictionary
        record = {"mse_nuh": [],
                  "mse_gt": [],
                  "fidelity_loss": [],
                  "cpu_time": [],
                  "SER": []
                  }
    
        results = None
        for t in range(num_iter):
            # Run DIP
            Y_hat = G(z)
            fidelity_loss = fn(Y_hat,b)
    
            total_loss = fidelity_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()
    
    
            if results is None:
                results = Y_hat.detach().numpy().reshape((-1,1), order='F')
            else:
                results = results * 0.99 + Y_hat.detach().numpy().reshape((-1,1), order='F') * 0.01
            Y_hat_numpy = Y_hat.detach().numpy().reshape((-1,1), order='F')
            
            # Measure
            X_hat = np.dot(Hinv,Y_hat_numpy)
            X_hat_results = np.dot(Hinv, results)
            
            mse_nuh = np.mean((X_hat - X[bs]) ** 2)
            mse_gt = np.mean((X_hat_results - X[bs]) ** 2)
            ser = CalcSER(X[bs], X_hat, M)
            # fidelity_loss = fn(torch.tensor(results).cuda()).detach()
            # fidelity_loss = fn(torch.tensor(results)).detach()
    
            # Record the result
            record["mse_nuh"].append(mse_nuh)
            record["mse_gt"].append(mse_gt)
            record["fidelity_loss"].append(fidelity_loss.item())
            record["SER"].append(ser)
            record["cpu_time"].append(time.time())
            if (t + 1) % 100 == 0:
                print('Batch %3d: Iter %5d   MSE_nuh: %e MSE_gt: %e  SER: %e' % (bs+1, t + 1, mse_nuh, mse_gt, ser))
        # Looking its best stopping point
        minvalue_nuh = min(record["mse_nuh"])
        minvalue_gt = min(record["mse_gt"])
        bsp_nuh = record["mse_nuh"].index(minvalue_nuh)
        bsp_gt = record["mse_gt"].index(minvalue_gt)
        optimser_nuh = record["SER"][bsp_nuh]
        optimser_gt = record["SER"][bsp_gt]
        
        # Save the process
        np.savez(results_dir+'record %0.2f batch %2d BSP(%d, %d) value(%e, %e) SER(%e, %e)' % (SNR_dB,bs,
                                                                                               bsp_nuh,bsp_gt,
                                                                                               minvalue_nuh,minvalue_gt,
                                                                                               optimser_nuh,optimser_gt), **record)
        plt.plot(np.arange(50000),record["SER"])

# Create fidelity loss function
def fn(x,b): 
    return torch.norm(x.reshape(-1) - b) ** 2 / 2


# Main program
tic = time.time()
# Check if cuda is available or not
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
    
datasets_dir = './data/'
if not os.path.isdir(datasets_dir):
    os.makedirs(datasets_dir)
results_dir = './data/results/'

# Create SNR_dB array
arrSNR_dB = np.linspace(0,30,3)
for SNR_dB in arrSNR_dB:
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    run(num_iter=50000,
        batch_size=10, 
        Nt=8, Nr=8, M=4,
        GD_lr=0.001, 
        SNR_dB=SNR_dB, 
        results_dir = results_dir,
        datasets_dir=datasets_dir,
        savedataset=True,
        seed=42)
toc = time.time()
print(f'Time spent by the system: {toc-tic}')