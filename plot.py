# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:47:48 2020

@author: wyue
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def readExcel(path,cols=[2,3]):
    df_array = []
    for file in os.listdir(path): 
        ## Ignore other file types 
        if file[-5:] != ".xlsx" or file[0:2] == '~$':
            continue

        ## Store csv file in a Pandas DataFrame. Only read 'Angle' and 'Torque'
        df = pd.read_excel(path + file,usecols=cols)
        df_array.append(df.to_numpy())
    return df_array

def plot_torque(torques):
    #plt.figure(figsize=(14,10))
    lgd = ['','2','3']
    for i,t in enumerate(torques):
        plt.plot(-t[:,1],-t[:,0],linewidth=linewidth,label=i)
        #plt.xlabel('Angle(degree)', fontsize=10)
        #plt.ylabel('Torque(Nm)', fontsize=10)
        #plt.legend()
    plt.savefig('TimeSeries.png',dpi=400)

def plot_erors_compare(data,colors):
    m = len(data)
    maxv = np.max(data) + 0.02
    fig, axs = plt.subplots(1, m,figsize=(12,3))
    for i in range(m):
        axs[i].plot(data[i], '-x',color=colors[i])
        axs[i].set_ylim(0, maxv)
        axs[i].grid(True)
    plt.savefig('compare_old_new.png',dpi=400)

cwd_up = os.path.dirname(os.getcwd())

## Global variables for plot
legend_fontsize = 8
figure_width = 4
figure_hight = 3
linewidth = 1


#path = cwd_up+"/data/plot/"
#data = readExcel(path)
#plot_torque(data)

path = cwd_up+"/data/plot/old_new_compare/"
data = readExcel(path,cols=[0])
plot_erors_compare(data,['c','y'])