# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:52:34 2020

@author: wyue

User FFT to filter noise
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def readData(file):
    ## Store csv file in a Pandas DataFrame. Only read 'Id' and 'Torque'
    df = pd.read_excel(file)
    df = df.to_numpy()
    return df[0:280]




f = readData("Torque.xlsx")
#f = f.squeeze(1)
"""
cwd_up = os.path.dirname(os.getcwd())
data_dir = cwd_up + "./data/test/VL/valid/"

f = np.load(data_dir+'sequence.npy')
f = f.squeeze(0)
f = f.squeeze(1)
"""
plt.figure()
plt.plot(f[0:280],color='c',LineWidth=1.5)

zeros = np.zeros((1000,1))
f = np.concatenate([f,zeros])

n = len(f)
fhat = np.fft.fft(f,n,axis=0)

## Power Spectrum Density
PSD = fhat * np.conj(fhat) / n 

freq = np.arange(n)

L = np.arange(1,np.floor(n/2),dtype='int')
plt.figure()
plt.plot(freq[L],PSD[L],color='c',LineWidth=2,label='PSD')


#mask = (1+np.cos(np.arange(n)*np.pi/n*2))/2
#fhat = fhat*mask

#fhat[100:1179] = 0
fhat[100:1181] = 0
#fhat[20:261] = 0
ffilt = np.fft.ifft(fhat,axis=0) # Inverse FFT for filtered time signal

plt.figure()
plt.plot(f[:240],color='c',LineWidth=1.5)
plt.plot(ffilt[:240],color='r',LineWidth=1.5)





