import os
from tools import *
import numpy as np
import torch
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt
import time
from models import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def run(num_iter=100000, trial_size=100, 
        Nt=8, Nr=8, M=16, GD_lr=0.001, SNR_dB=10, 
        results_dir=None, datasets_dir=None, savedataset=False, writer=None, ser=dict(),
        seed=42):
    # Generate X, Y, H using random function or from dataset
    (X, Y, H) = generate(trial_size, Nt, Nr, M, SNR_dB, seed, rootdir=datasets_dir, save=savedataset)
    # (X, Y, H) = pd.read_excel('users.xlsx', sheet_name = [0,1,2])
    
    # Create fidelity loss function
    fn = torch.nn.MSELoss()
    mat_xhat = np.array([])
    mat_xtrue = np.array([])
    for ts in range(trial_size):
        b_numpy = Y[ts].reshape((1,1,-1,2),order='F')
        b = torch.from_numpy(b_numpy).type(dtype)#.reshape(-1)
        
        Ht = H[ts].transpose(1,0)
        Hinv = np.dot(np.linalg.inv(np.dot(Ht, H[ts])), Ht)
        
        G = skip(1, 1,
                 num_channels_down=[4, 4, 8, 8],
                 num_channels_up=[4, 4, 8, 8],#[16, 32, 64, 128, 128],
                 num_channels_skip=[0, 0, 0, 0],
                 filter_size_up=3, filter_size_down=3, filter_skip_size=1,
                 upsample_mode='nearest',  # downsample_mode='avg',
                 need1x1_up=False,
                 need_sigmoid=True, need_bias=True, pad='zero', act_fun='LeakyReLU').type(dtype)
        G.apply(weight_reset)
        z = torch.zeros_like(torch.empty(1,1,Nr,2)).type(dtype).normal_()
    
        z.requires_grad = False
        opt = optim.Adam(G.parameters(), lr=GD_lr)
        
        # Record dictionary
        record = {"mse_yhat_y": [],
                  "mse_gt": [],
                  "fidelity_loss": [],
                   "cpu_time": [] # ,"SER": []
                  }
    
        results = None
        result_x = None
        for t in range(num_iter):
            # Run DIP
            Y_hat = (G(z) - 0.5) * np.sqrt(M)
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
            
            mse_yhat_y = np.mean((Y_hat_numpy - Y[ts]) ** 2)
            mse_gt = np.mean((X_hat_results - X[ts]) ** 2)
            # fidelity_loss = fn(torch.tensor(results).cuda()).detach()
            # fidelity_loss = fn(torch.tensor(results)).detach()    
            # Record the result
            record["mse_yhat_y"].append(mse_yhat_y)
            record["mse_gt"].append(mse_gt)
            record["fidelity_loss"].append(fidelity_loss.item())
            # record["SER"].append(ser)
            record["cpu_time"].append(time.time())
            if (t + 1) % 100 == 0:
                print('trial %3d: Iter %5d   MSE_yhat_y: %e MSE_gt: %e' % (ts+1, t + 1, mse_yhat_y, mse_gt))
                # print('trial %3d: Iter %5d   MSE_yhat_y: %e MSE_gt: %e' % (ts+1, t + 1, mse_yhat_y, mse_gt))
            
            if (result_x is not None and np.mean(np.abs(result_x-X_hat)) < 0.0005):
                break
            else:
                result_x = X_hat
                
        # Looking its best stopping point
        minvalue_nuh = (record["mse_yhat_y"][t])
        minvalue_gt = (record["mse_gt"][t])
        bsp_nuh = record["mse_yhat_y"].index(minvalue_nuh)
        bsp_gt = record["mse_gt"].index(minvalue_gt)
        # optimser_nuh = record["SER"][t]
        # optimser_gt = record["SER"][t]
        
        # Save the process
        writer = SaveRecord(record, writer, ts, SNR_dB, bsp_nuh, bsp_gt)
        # print('record %0.2f trial %2d SP(%d) value(%e, %e) SER(%e, %e)' % (SNR_dB,ts+1, 
        #                                                                     t,
        #                                                                     minvalue_nuh,minvalue_gt,
        #                                                                     optimser_nuh,optimser_gt))
        print('record %0.2f trial %2d BSP(%d, %d) value(%e, %e)' % (SNR_dB,ts+1, 
                                                                    bsp_nuh, bsp_gt,
                                                                    minvalue_nuh,minvalue_gt))

        plt.figure()
        plt.title("trial %d" % (ts+1))
        plt.plot(np.arange(t+1), record["mse_gt"])
        plt.plot(np.arange(t+1), record["mse_yhat_y"])
        plt.legend(["mse_gt", "mse_yhat_y"], loc="upper right")
    
        mat_xhat = np.append(mat_xhat, X_hat).reshape(-1,1)
        mat_xtrue = np.append(mat_xtrue, X[ts]).reshape(-1,1)
        
    ser["%d_dB" % SNR_dB] = CalcSER(mat_xtrue, mat_xhat, M)
    firstline = f'The SER of SNR_dB = {SNR_dB} is {ser["%d_dB" % SNR_dB]} dB.'
    print(firstline)
    fout = open(results_dir+"ser_%0.2f_dB.txt" % SNR_dB, 'w')
    fout.write(firstline)
    fout.close()
    writer.save()
    writer.close()
    

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
arrSNR_dB = np.array([30])
ser = dict()
for SNR_dB in arrSNR_dB:
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    writerdata = pd.ExcelWriter(results_dir + 'results_%0.2f_dB.xlsx' % SNR_dB, engine='xlsxwriter')
    run(num_iter=50000,
        trial_size=10000, 
        Nt=8, Nr=8, M=4,
        GD_lr=0.001, 
        SNR_dB=SNR_dB, 
        results_dir = results_dir,
        datasets_dir=datasets_dir,
        savedataset=True, writer=writerdata, ser = ser,
        seed=None)
toc = time.time()
print(f'Time spent by the system: {toc-tic}')