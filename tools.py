import numpy as np
import pandas as pd
import os, sys

def generate(batch_size, Nt, Nr, M, SNR_dB, seed='None', rootdir=None,save=False):
    
    # Look at the seed value
    if seed != 'None':
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    # Generate H matrix real-equivalent domain
    Hr = rng.randn(batch_size,Nr,Nt) * np.sqrt(0.5/Nr)
    Hi = rng.randn(batch_size,Nr,Nt) * np.sqrt(0.5/Nr)
    # H = [[Hr -Hi],
    #      [Hi  Hr]]
    H = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
    
    # Generate X (x in real-equivalent domain)
    L = np.int(np.sqrt(M))
    x = rng.randint(low=0,high=L,size=(batch_size,Nt)) + 1j*rng.randint(low=0,high=L,size=(batch_size,Nt))
    x = x*2-(1+1j)*(L-1)
    x = normsymbol(x, L)
    # change the x to real-equivalent domain
    X = np.concatenate((x.real, x.imag), axis=1)
    X = np.reshape(X,(batch_size,-1,1))
    
    # Generate Noise
    sigma2 =  (2 * Nt) / (np.power(10, SNR_dB/10) * (2*Nr))
    noise = np.sqrt(sigma2 / 2) * rng.randn(batch_size, 2*Nr,1)
    
    # Generate Y (y in real-equivalent domain)
    Y = np.matmul(H,X) + noise
    
    if save == False:
        return (X, Y, H)
    
    if rootdir == None:
        rootdir = './data/'
        if not os.path.isdir(rootdir):
            os.makedirs(rootdir)
    
    # Saving those matrices to help debugging later
    for fname in ['matX','matY','matH']:
        writer = pd.ExcelWriter(rootdir + '%s_%0.2f_dB.xlsx' % (fname, SNR_dB), engine='xlsxwriter')
    
        if fname == 'matX':
            data = X
        elif fname == 'matY':
            data = Y
        elif fname == 'matH':
            data = H
        else:
            break

        for i in range(0, data.shape[0]):
            df = pd.DataFrame(data[i,:,:])
            df.to_excel(writer, sheet_name='iter_%d' % i)

        writer.save()
    
    return (X, Y, H)

def normsymbol(x, L):
    constellation = np.linspace(int(-L+1), int(L-1), int(L))
    alpha = np.sqrt((constellation ** 2).mean())
    x /= (alpha * np.sqrt(2))
    
    return x