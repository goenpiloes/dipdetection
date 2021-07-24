#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:28:17 2021

@author: nuhardjowono
"""

import numpy as np

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
    Y = np.matmul(H,X) #+ noise
    
    return (X, Y, H)

def normsymbol(x, L):
    constellation = np.linspace(int(-L+1), int(L-1), int(L))
    alpha = np.sqrt((constellation ** 2).mean())
    x /= (alpha * np.sqrt(2))
    return x

(X, Y, H) = generate(batch_size=10, Nt=8, Nr=8, M=16, SNR_dB=10)

bs = 2

Ht = H[bs].transpose(1,0)
Hinv = np.dot(np.linalg.inv(np.dot(Ht, H[bs])), Ht)

X_hat = np.dot(Hinv,Y[bs])