# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:46:56 2020

@author: wyue
"""
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import pandas as pd
from sklearn import preprocessing
data = np.load('sequence.npy')

# torque = data[0,:,0]
# torque = torque.reshape(-1,1)
# min_max_scaler = preprocessing.MinMaxScaler()
# torque_new = min_max_scaler.fit_transform(torque)

# angle = data[0,:,1]
# angle = angle.reshape(-1,1)
# min_max_scaler_angle = preprocessing.MinMaxScaler()
# angle_new = min_max_scaler_angle.fit_transform(angle)






plt.subplot(4,1,1)
plt.plot(torque)

plt.subplot(4,1,2)
plt.plot(torque_new)

plt.subplot(4,1,3)
plt.plot(angle)

plt.subplot(4,1,4)
plt.plot(angle_new)


