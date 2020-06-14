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
"""
x = np.linspace(0, 10, 500)
y = np.sin(x)

plt.figure()

# Using set_dashes() to modify dashing of an existing line
plt.plot(x, y, label='Using set_dashes()')

plt.figure()
# Using plot(..., dashes=...) to set the dashing when creating a line
plt.plot(x, y - 0.2, dashes=[6, 2], label='Using the dashes parameter')


plt.show()
"""
dic = {}
dic['a'] = 1
