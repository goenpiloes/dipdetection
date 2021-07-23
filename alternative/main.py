import os, sys, glob
from tools import *
import numpy as np
import torch
from torch import optim
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

    # Create model G and random noise input z
    G = skip(1, 1,
             num_channels_down=[2],
             num_channels_up=[2],#[16, 32, 64, 128, 128],
             num_channels_skip=[0],
             filter_size_up=2, filter_size_down=2, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    z = torch.zeros_like(torch.empty(1,1,Nr,2)).type(dtype).normal_()

    z.requires_grad = False
    opt = optim.Adam(G.parameters(), lr=GD_lr)
    
    # Record dictionary
    record = {"mse_nuh": [],
              "mse_gt": [],
              "stopping_point": [],
              "cpu_time": [],
              }

    results = None
    for bs in range(batch_size):
        for t in range(num_iter):
            b = Y[bs]
            # Run DIP
            Y_hat = G(z)
            fidelity_loss = fn(Y_hat,b)
    
            total_loss = fidelity_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()
    
    
            if results is None:
                results = Y_hat.detach().cpu().numpy()
            else:
                results = results * 0.99 + Y_hat.detach().cpu().numpy() * 0.01
            
            # Measure
            with torch.no_grad():
                mse_nuh = np.mean((Y_hat.cpu().numpy() - b) ** 2)
                mse_gt = np.mean((Y_hat.cpu().numpy() - results) ** 2)
            # fidelity_loss = fn(torch.tensor(results).cuda()).detach()
            # fidelity_loss = fn(torch.tensor(results)).detach()
    
            
            # With linf_ball_projection
            # # for x
            # with torch.no_grad():
            #     x = linf_proj(Gz.detach() - scaled_lambda_, b, noise_sigma)
            #     # x = Gz.detach() - scaled_lambda_
    
            # # for z (GD)
            # opt_z.zero_grad()
            # Gz = G(z)
            # loss_z = torch.norm(b- Gz) ** 2 / 2 + (rho / 2) * torch.norm(x - G(z) + scaled_lambda_) ** 2
            # loss_z.backward()
            # opt_z.step()
    
            # # for dual var(lambda)
            # with torch.no_grad():
            #     Gz = G(z).detach()
            #     x_Gz = x - Gz
            #     scaled_lambda_.add_(sigma_0 * x_Gz)
    
            # if results is None:
            #     results = Gz.detach()
            # else:
            #     results = results * 0.99 + Gz.detach() * 0.01
    
            # psnr_gt = peak_signal_noise_ratio(x_true.cpu().numpy(), results.cpu().numpy())
            # mse_gt = np.mean((x_true.cpu().numpy() - results.cpu().numpy()) ** 2)
            # fidelity_loss = fn(torch.tensor(results).cuda()).detach()
    
            # Record the result
            record["mse_nuh"].append(mse_nuh)
            record["mse_gt"].append(mse_gt)
            record["stopping_point"].append(fidelity_loss.item())
            record["cpu_time"].append(time.time())
            if (t + 1) % 10 == 0:
                print('Batch %3d: Iteration %5d   MSE_nuh: %e MSE_gt: %e' % (batch_size+1, t + 1, mse_nuh, mse_gt))
        np.savez(results_dir+'record %0.2f batch %2d' % (SNR_dB,batch_size), **record)

# Create fidelity loss function
def fn(x,b): 
    return torch.norm(x.reshape((-1),order='F') - b) ** 2 / 2


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
arrSNR_dB = np.linspace(0,15,4)
for SNR_dB in arrSNR_dB:
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    run(num_iter=10000,
        batch_size=100, 
        Nt=8, Nr=8, M=16,
        GD_lr=0.001, 
        SNR_dB=SNR_dB, 
        results_dir = results_dir,
        datasets_dir=datasets_dir,
        savedataset=True,
        seed=42)
toc = time.time()
print(f'Time spent by the system: {toc-tic}')