import os, sys, glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tools import *
import numpy as np
import torch
import time
from models import *

def run(batch_size, ):
    # Generate X, Y, H using random function or from dataset
    (X, Y, H) = generate(batch_size=10, Nt=3, Nr=2, M=16, SNR_dB=10, seed=42)
    # (X, Y, H) = pd.read

    # Create model G and random noise input z
    G = skip(3, 3,
             num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],#[16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    z = torch.zeros_like(x_true).type(dtype).normal_()

    z.requires_grad = False
    opt = optim.Adam(G.parameters(), lr=GD_lr)
    
    # Create fidelity loss function
    def fn(x): return torch.norm(x.reshape(-1) - b) ** 2 / 2
    
    # Record dictionary
    record = {"psnr_gt": [],
              "mse_gt": [],
              "total_loss": [],
              "prior_loss": [],
              "fidelity_loss": [],
              "cpu_time": [],
              }

    results = None
    for t in range(num_iter):
        x = G(z)
        fidelity_loss = fn(x)

        # prior_loss = (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        total_loss = fidelity_loss #+ 0.01 * prior_loss
        opt.zero_grad()
        total_loss.backward()
        opt.step()


        if results is None:
            results = x.detach().cpu().numpy()
        else:
            results = results * 0.99 + x.detach().cpu().numpy() * 0.01

        psnr_gt = peak_signal_noise_ratio(x_true.cpu().numpy(), results)
        mse_gt = np.mean((x_true.cpu().numpy() - results) ** 2)
        # fidelity_loss = fn(torch.tensor(results).cuda()).detach()

        if (t + 1) % 1000 == 0:
            if num_channels == 3:
                imsave(specific_result_dir + 'iter%d_PSNR_%.2f.png'%(t, psnr_gt), results[0].transpose((1,2,0)))
            else:
                imsave(specific_result_dir + 'iter%d_PSNR_%.2f.png'%(t, psnr_gt), results[0, 0], cmap='gray')


        record["psnr_gt"].append(psnr_gt)
        record["mse_gt"].append(mse_gt)
        record["fidelity_loss"].append(fidelity_loss.item())
        record["cpu_time"].append(time.time())
        if (t + 1) % 10 == 0:
            print('Img %d Iteration %5d   PSRN_gt: %.2f MSE_gt: %e' % (f_num, t + 1, psnr_gt, mse_gt))
    np.savez(specific_result_dir+'record', **record)
tic = time.time()
run()
toc = time.time()
print(f'Time spent by the system: {toc-tic}')