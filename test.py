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


x = torch.tensor([1.0,2.0], requires_grad=True)

print(x)

y1 = torch.pow(x[0],2) + torch.pow(x[1],2)
print(y1)


#xx = torch.tensor([1.0,2.0], requires_grad=True)
xx = x.clone()
y2 = xx[0]*xx[1]
print(y2)


y1.backward(retain_graph=True)
print('gradient of y1 with respect to x',x.grad)

y2.backward(retain_graph=True)
print('gradient of y2 with respect to x',xx.grad)

y = y1+y2
y.backward()
print(y.grad)
#y = y1+y2
#y.backward()
#print('gradient of y with respect to x',x.grad)


"""
import torch
from torch.autograd import Variable

def basic_fun(x):
    return 3*(x*x)

def get_grad(inp, grad_var):
    A = basic_fun(inp)
    A.backward()
    return grad_var.grad

x = Variable(torch.FloatTensor([1]), requires_grad=True)
xx = x.clone()

# Grad wrt x will work
print(x.grad_fn is None) # is it a leaf? Yes
print(get_grad(x, x))
print(get_grad(xx, x))

# Grad wrt xx won't work
print(xx.grad_fn is None) # is it a leaf? No
print(get_grad(xx, xx))
print(get_grad(x, xx))
"""