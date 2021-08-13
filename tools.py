import numpy as np
import pandas as pd
import os

def generate(trial_size, Nt, Nr, M, SNR_dB, seed='None', rootdir=None,save=False):
    '''

    Parameters
    ----------
    trial_size : int
        number of simulations performed in a single SNR value.
    Nt : int
        number of transmitter antennas.
    Nr : int
        number of receiver antennas .
    M : int
        Modulation orde.
    SNR_dB : float
        signal-to-noise (SNR) value in dB.
    seed : int, optional
        initialize a pseudorandom number generator. The default is 'None'.
    rootdir : string, optional
        path to save the X, Y, H dataset. The default is None.
    save : bool, optional
        save the dataset. The default is False.

    Returns
    -------
    X : numpy array of float
        transmitted symbols in real-equivalent domain.
    Y : numpy array of float
        received symbols in real-equivalent domain.
    H : numpy array of float
        rayleigh channel in real-equivalent domain.
    noisevar : float
        noise variance.

    '''
    # Look at the seed value
    if seed != 'None':
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    # Generate H matrix real-equivalent domain
    Hr = rng.randn(trial_size,Nr,Nt) * np.sqrt(0.5/Nr)
    Hi = rng.randn(trial_size,Nr,Nt) * np.sqrt(0.5/Nr)
    # H = [[Hr -Hi],
    #      [Hi  Hr]]
    H = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
    
    # Generate X (x in real-equivalent domain)
    L = np.int(np.sqrt(M))
    x = rng.randint(low=0,high=L,size=(trial_size,Nt)) + 1j*rng.randint(low=0,high=L,size=(trial_size,Nt))
    x = x*2-(1+1j)*(L-1)
    x = normsymbol(x, L)
    # change the x to real-equivalent domain
    X = np.concatenate((x.real, x.imag), axis=1)
    X = np.reshape(X,(trial_size,-1,1))
    
    # Generate Noise
    sigma2 =  (2 * Nt) / (np.power(10, SNR_dB/10) * (2*Nr))
    noise = np.sqrt(sigma2 / 2) * rng.randn(trial_size, 2*Nr,1)
    
    # Generate Y (y in real-equivalent domain)
    Y = np.matmul(H,X) + noise
    
    if save == False:
        return (X, Y, H, sigma2/2)
    
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
            df.to_excel(writer, sheet_name='iter_%d' % (i+1))

        writer.save()
    
    return (X, Y, H, sigma2/2)

def normsymbol(x, L):
    '''

    Parameters
    ----------
    x : numpy array of float
        unnormalized symbols.
    L : int
        length of its constellation diagram. L is equal to square root of modulation orde M (L = sqrt(M)).

    Returns
    -------
    x : numpy array of float
        normalized symbols.

    '''
    constellation = np.linspace(int(-L+1), int(L-1), int(L))
    alpha = np.sqrt((constellation ** 2).mean())
    x /= (alpha * np.sqrt(2))
    
    return x

def CalcSER(xtrue, xhat, M):
    '''

    Parameters
    ----------
    xtrue : numpy array of float
        estimated symbols in real-equivalent domain.
    xhat : numpy array of float
        true symbols in real-equivalent domain.
    M : int
        modulation orde.

    Returns
    -------
    BER : float
        Bit error rate (BER).

    '''
    xtrue = xtrue.reshape((-1,1))
    xhat = xhat.reshape((-1,1))
    L = int(np.sqrt(M))
    
    # Make reference constellation points
    cons = np.array([[i for i in range(-L+1, L, 2)]],dtype=float)
    cons = normsymbol(cons, L)
    
    # Calculate errors of yhat
    errors = np.abs(xhat - cons)**2
    xhat_idx = errors.argmin(axis=1)
    xhat_idx = xhat_idx.reshape((-1,1))
    
    # Calculate the bit of ytrue
    lookzero = abs(xtrue - cons)**2
    xtrue_idx = lookzero.argmin(axis=1)
    xtrue_idx = xtrue_idx.reshape((-1,1))
    
    # SER Calculation
    return np.mean(np.not_equal(xtrue_idx, xhat_idx))

def SaveRecord(d, writer, trial, t=0):
    '''

    Parameters
    ----------
    d : TYPE
        DESCRIPTION.
    writer : TYPE
        DESCRIPTION.
    trial : TYPE
        DESCRIPTION.
    t : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    writer : TYPE
        DESCRIPTION.

    '''
    df = pd.DataFrame(d)
    df.to_excel(writer, sheet_name='trial_%d best iter = %d' % (trial+1,t))
    
    return writer