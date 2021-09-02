# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 23:15:23 2021

@author: Nuh Hardjowono
"""

import pandas as pd
import numpy as np

xl = pd.ExcelFile("./data/test5/results2/results_30.00_dB.xlsx")
columns = None
arrMSE_last = np.array([])
arrSER_last = np.array([])
for idx, name in enumerate(xl.sheet_names):
    # print(f'Reading sheet #{idx}: {name}')
    sheet = xl.parse(name)
    # if idx == 0:
    #     columns = sheet.columns
    # sheet.columns = columns
    last_MSE = sheet['mse_xhat_x'].iloc[299]
    last_SER = sheet['SER'].iloc[299]
    arrMSE_last = np.append(arrMSE_last, last_MSE)
    arrSER_last = np.append(arrSER_last, last_SER)

MSE_avg = np.mean(arrMSE_last)
SER_avg = np.mean(arrSER_last)
# MSE_median = np.median(arrMSE_last)
# MSE_var = np.var(arrMSE_last)
print(MSE_avg)
print(SER_avg)
# print(MSE_median)
# print(MSE_var)
    