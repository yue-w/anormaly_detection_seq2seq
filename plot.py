# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:47:48 2020

@author: wyue
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

def readExcel(path):
    df_array = []
    for file in os.listdir(path): 
        ## Ignore other file types 
        if file[-5:] != ".xlsx" or file[0:2] == '~$':
            continue

        ## Store csv file in a Pandas DataFrame. Only read 'Angle' and 'Torque'
        df = pd.read_excel(path + file,usecols=[2,3])
        df_array.append(df.to_numpy())
    return df_array

def plot_torque(torques):
    #plt.figure(figsize=(14,10))
    lgd = ['','2','3']
    for i,t in enumerate(torques):
        plt.plot(-t[:,1],-t[:,0],linewidth=1,label=i)
        #plt.xlabel('Angle(degree)', fontsize=10)
        #plt.ylabel('Torque(Nm)', fontsize=10)
        #plt.legend()
    plt.savefig('TimeSeries.png',dpi=400)
    
cwd_up = os.path.dirname(os.getcwd())


path = cwd_up+"/data/plot/"

data = readExcel(path)

plot_torque(data)