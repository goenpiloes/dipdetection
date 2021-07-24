#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:28:17 2021

@author: nuhardjowono
"""

import numpy as np

x = np.random.randn(1,4)
print(f'x = \n {x}')
x1 = x.reshape((-1,2),order='F')
print(f'x1 = \n {x1}')