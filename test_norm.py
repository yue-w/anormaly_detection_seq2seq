# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:44:15 2020

@author: wyue
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

y = np.array([[1,2,3.11,4.2],[1,2,3.45,4]])*1.0
yhat = np.array([[1.1,2,3,4],[1,2,3,4]])*1.0

#y = np.random.randn(2,4)
#yhat = np.random.randn(2,4)

loss = nn.MSELoss(reduction='sum')
y_tr = torch.from_numpy(y)
yhat_tr = torch.from_numpy(yhat)
loss_nn = loss(y_tr, yhat_tr)
print('torch:',loss_nn)


loss = np.linalg.norm(y-yhat)
print('numpy:',loss**2)

yy = (y-yhat)
print('new:', np.sum(np.power(yy,2)))

norm = np.linalg.norm(y-yhat,axis=0)
print('Norm of each time step',norm)


