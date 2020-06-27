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
import pandas.ewma as ewma


 
# make a hat function, and add noise
x = np.linspace(0,1,100)
x = np.hstack((x,x[::-1]))
x += np.random.normal( loc=0, scale=0.1, size=200 )
plot( x, alpha=0.4, label='Raw' )
 
# take EWMA in both directions with a smaller span term
fwd = ewma( x, span=15 ) # take EWMA in fwd direction
bwd = ewma( x[::-1], span=15 ) # take EWMA in bwd direction
c = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
c = np.mean( c, axis=0 ) # average
 
# regular EWMA, with bias against trend
plot( ewma( x, span=20 ), 'b', label='EWMA, span=20' )
 
# "corrected" (?) EWMA
plot( c, 'r', label='Reversed-Recombined' )
 
legend(loc=8)
savefig( 'ewma_correction.png', fmt='png', dpi=100 )

