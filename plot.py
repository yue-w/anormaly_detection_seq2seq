# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:08:19 2020

@author: wyue
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


cwd_up = os.path.dirname(os.getcwd())
file = cwd_up+"/data/test/ttt_trace.xlsx"
df = pd.read_excel(file,usecols=[2,3])
ta = df.to_numpy()

plt.figure()
x = np.arange(599)
plt.scatter(x, ta[:599,0],s=1)

plt.savefig("torque.png",dpi=400)


#plt.figure()
#plt.plot(ta[:599,1], ta[:599,0])