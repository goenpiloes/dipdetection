# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 23:15:23 2021

@author: Nuh Hardjowono

This file is only useful for calculating MSE from main_old result.
"""

import pandas as pd
import numpy as np

xl = pd.ExcelFile("./new/results/results_0.00_dB.xlsx")

arrMSE_last = np.array([])
for idx, name in enumerate(xl.sheet_names):
    sheet = xl.parse(name)
    last_MSE = sheet['mse_xhat_x'].iloc[-1]
    arrMSE_last = np.append(arrMSE_last, last_MSE)

MSE_avg = np.mean(arrMSE_last)
MSE_median = np.median(arrMSE_last)
MSE_var = np.var(arrMSE_last)
print(MSE_avg)
print(MSE_median)
print(MSE_var)