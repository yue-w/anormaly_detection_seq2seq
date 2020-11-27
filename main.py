# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1  2020

@author: Yue Wang

Some of the ideas/codes come from 
1. https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
2. https://blog.floydhub.com/gru-with-pytorch/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import TensorDataset, DataLoader
import time
import os
from tqdm import tqdm_notebook
import copy
import torch.nn.functional as F
from sklearn import preprocessing


class EngineData:
    def __init__(self,path):
        self.path = path
        self.num_engines = -1
        
    def readExcel(self,col=[0,2]):
        df_array = []
        for file in tqdm_notebook(os.listdir(self.path)): 
            ## Ignore other file types 
            if file[-5:] != ".xlsx" or file[0:2] == '~$':
                continue

            ## Store csv file in a Pandas DataFrame. Only read 'Id' and 'Torque', and 'Angle'
            df = pd.read_excel(self.path + file,usecols=col)
            df_array.append(df)
        
        self.rawdata = pd.concat(df_array)
        
    def writeToCSV(self):
        self.rawdata.to_csv(self.path+'alldata.csv',index=False)

        
    def getNumOfEngines(self):
        N = 0
        for i,row in self.rawdata.iterrows():
            if row['Id'] == 0:
                N+=1

        self.num_engines = N

    def getSequence(self,seq_len,readAngle=False):
        if self.num_engines<0:
            self.getNumOfEngines()
            
            ## Determine the length of the data
            if seq_len[1]<seq_len[2]:## This is used for TTT. A sequence of data is deleted.
                leng = seq_len[0]+seq_len[2]-seq_len[1]-1
            else:
                leng = seq_len[0]
        if readAngle == False:
            self.sequence = np.zeros((self.num_engines,leng,1))
        else:
            self.sequence = np.zeros((self.num_engines,leng,2))
        
        ## Set values
        index = -1
        for i,row in self.rawdata.iterrows():
            id = int(row['Id'])
            if id<seq_len[0] or (id>seq_len[1] and id<seq_len[2]):
                if id==0:
                    index+=1
                    idx = 0
                if readAngle == False:
                    self.sequence[index][idx] = row['Torque']
                else:
                    self.sequence[index][idx] = row[['Torque','Angle']]
                idx +=1

    def preprocess(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        for i in range(len(self.sequence)):
            # plt.subplot(4,1,1)
            # plt.plot(self.sequence[i][:,0])
            # plt.subplot(4,1,2)
            # plt.plot(self.sequence[i][:,1])
            self.sequence[i] = min_max_scaler.fit_transform(self.sequence[i])
            # plt.subplot(4,1,3)
            # plt.plot(self.sequence[i][:,0])
            # plt.subplot(4,1,4)
            # plt.plot(self.sequence[i][:,1])
            # return
## Global variables for plot
legend_fontsize = 8
figure_width = 4
figure_hight = 3


def plot_torque(angle,torques):
    plt.figure(figsize=(figure_width,figure_hight))
    plt.plot(angle,torques, color="g")
    plt.ylabel('Torque')
    plt.legend()


def plot_pred_true(pred,truth,figname):
    plt.figure(figsize=(figure_width,figure_hight))
    plt.plot(pred, "-x", color="g", label="Reconstructed",linewidth=1,markersize=4)
    plt.plot(truth, color="b", label="Original")
    #plt.ylabel('Torque(N.m)')
    #plt.xlabel('Angle(degree')
    plt.legend(fontsize=legend_fontsize) # ,loc='lower left'
    plt.xticks(fontsize=legend_fontsize)
    plt.yticks(fontsize=legend_fontsize) #rotation=90
    plt.savefig(figname,dpi=400)

def plot_error(error,figname,color='c'):
    plt.figure(figsize=(figure_width,figure_hight))
    plt.plot(error, "-x", color=color, markersize=4)
    #plt.ylabel('Error')
    #plt.legend()
    plt.xticks(fontsize=legend_fontsize)
    plt.yticks(fontsize=legend_fontsize) #rotation=90
    plt.savefig(figname,dpi=400)

def error_limit(error_list):
    upper = 0
    lower = 0
    max_list = []
    min_list = []
    for errors in error_list:
        max_list.append(np.amax(errors,axis=0))
        min_list.append(np.amin(errors,axis=0))

    upper = max(max_list)
    lower = min(min_list)
    return upper, lower 



class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers,batch_first=True, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        inputs = self.dropout(inputs)
        outputs, (hidden, cell) = self.rnn(inputs)
        
        return outputs, hidden, cell
    """
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hid_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hid_dim).zero_().to(device))
        return hidden
    """

class EncoderATT(nn.Module):
    """
    Encoder for the Attention model
    """
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, dropout, doubledecoder=False):
        super().__init__()
        self.n_layers = 1
        self.hid_dim = enc_hid_dim
        self.rnn = nn.LSTM(input_dim, enc_hid_dim,batch_first=True, dropout=dropout, bidirectional=True)
        ## The following two linear layer is used to generate the initial 
        ## h and c that will feed into the decoder.
        ## Is the linear layer necessary? Try feeding 0 as initial input.
        self.fc_h_backward = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.fc_c_backward = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_h_forward = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.fc_c_forward = nn.Linear(enc_hid_dim, dec_hid_dim)
        if doubledecoder==True:
            self.doubledecoder = True
        else:
            self.doubledecoder = False
        
    def forward(self, inputs):
        inputs = self.dropout(inputs)
        outputs, (hidden, cell) = self.rnn(inputs)
        ## Pass the last hidden and cell of the backward into a linear, the output
        ## of the linear layer is used as the input hidden and cell for the decoder (attention)
        hidden_backward = torch.tanh(self.fc_h_backward(hidden[1,:,:])) ## Forward is indexed 0, backward is indexed 1
        cell_backward = torch.tanh(self.fc_c_backward(cell[1,:,:]))
        hidden_backward = hidden_backward.unsqueeze(1)
        cell_backward = cell_backward.unsqueeze(0)
        ## If this is the encoder for one decoder, the backward h,and c are returned
        if self.doubledecoder==False:
            return outputs,hidden_backward, cell_backward
        ## If this is the encoder for double decoder, both the forward and backward h and c are returned
        else:
            ## Pass the last hidden and cell of the backward into a linear, the output
            ## of the linear layer is used as the input hidden and cell for the decoder (attention)
            hidden_forward = torch.tanh(self.fc_h_forward(hidden[0,:,:])) ## Forward is indexed 0, backward is indexed 1
            cell_forward = torch.tanh(self.fc_c_forward(cell[0,:,:]))
            hidden_forward = hidden_forward.unsqueeze(1)
            cell_forward = cell_forward.unsqueeze(0)
            return outputs,hidden_backward,hidden_forward, cell_backward,cell_forward

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        
        attenDim = dec_hid_dim ##attenDim can be different from dec_hid_dim
        
        self.v = nn.Linear(attenDim, 1, bias = False)
        
        self.Wa = nn.Linear(dec_hid_dim, attenDim)
        self.Ua = nn.Linear(enc_hid_dim*2, attenDim)
        
    def forward(self, hidden, encoder_outputs):        
        src_len=encoder_outputs.shape[1]
        hidden = self.Wa(hidden)
        hidden = hidden.repeat(1,src_len,1)
        encoder_outputs = self.Ua(encoder_outputs)
        encoder_outputs = torch.tanh(encoder_outputs + hidden)
        energy = self.v(encoder_outputs)
        return F.softmax(energy, dim=1)
    
class OutputNetwork(nn.Module):
    def __init__(self,hid_dim,output_dim):
        super().__init__()
        
        self.dim1 = hid_dim
        self.dim2 = int(self.dim1/8)
        self.dim3 = int(self.dim2/2)
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.dim1, self.dim2)
        self.fc2 = nn.Linear( self.dim2, self.dim3)
        self.fc3 = nn.Linear(self.dim3, self.output_dim)
        self.relu = nn.ReLU()
    
    
    def forward(self,input):
        input = self.relu(input.squeeze(1))## Is the relu necessary?
        input = self.relu(self.fc1(input))
        input = self.relu(self.fc2(input))
        prediction = self.fc3(input)
        ## Try adding a second linear layer

        return prediction

class Decoder(nn.Module):
    def __init__(self,outputNet,input_dim, hid_dim,output_dim, n_layers, dropout):
        super().__init__()
        self.outputNet = outputNet
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers,batch_first=True, dropout = dropout)
        
    def forward(self, inputs, hidden, cell):
        inputs = inputs.unsqueeze(1)
        #inputs = self.dropout(inputs)
        output, (hidden, cell) = self.rnn(inputs, (hidden, cell)) 

        prediction = self.outputNet(output)
        ## Try adding a second linear layer

        return prediction, hidden, cell


class DecoderATT(nn.Module):
    def __init__(self,outputNet,attention, output_dim, enc_hid_dim, hid_dim, dropout):
        super().__init__()
        self.attention = attention
        self.outputNet = outputNet
        
        self.n_layers = 1
        self.output_dim = output_dim

        self.hid_dim = hid_dim
        self.dec_input_dim = self.output_dim+enc_hid_dim*2

        #self.linear_hid_dim2 = int(self.hid_dim/2)
        ## Concate the output of the rnn cell of the former step with the contex vector,
        ## and feed it into the rnn cell as input
        self.rnn = nn.LSTM(self.dec_input_dim, self.hid_dim, batch_first=True) 
        ## Feed the output/hidden layer of the rnn cell into linear layers

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden,cell, encoder_outputs):

        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.permute(0,2,1)
        
        context = torch.bmm(a, encoder_outputs)
        
        input = input.unsqueeze(1)
        input_context = torch.cat((input, context), dim = 2)
        #input_context = input_context.permute((1,0,2))
        #input_context = [1, batch size, (enc hid dim * 2) + emb dim]
        
        hidden = hidden.permute(1,0,2)
        output, (hidden, cell) = self.rnn(input_context, (hidden,cell))
        
        prediction = self.outputNet(output)
        
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        #self.reverse = reverse
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0]
        ## Output length
        tq_len = trg.shape[1]
        ## Output dimension (1)
        tq_dim = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size,tq_len,tq_dim).to(self.device)
        
        # Get the last hidden state of the encoder, 
        ## They will be used as the initial hidden state of the decoder
        src = src.to(self.device).float()
        encoder_outputs, hidden, cell = self.encoder(src)
        
        ## Initial input is always the 'gruond truth value'
        #inputs = copy.deepcopy(trg[:,0,:])
        ## Initial input is all zeros
        inputs = torch.zeros(trg[:,0,:].shape).to(self.device)
        ## The initial output is always the 'ground truth value'
        #outputs[:,0,:] = inputs
        for t in range(0, tq_len):
            ## if decoder has attention model, Use attention
            if hasattr(self.decoder, 'attention'): 
                output, hidden,cell = self.decoder(inputs.to(self.device).float(), hidden,cell, encoder_outputs)
                hidden = hidden.permute(1,0,2)
            ## Ohterwise, do not use attention
            else: 
                output, hidden, cell = self.decoder(inputs.to(self.device).float(), hidden, cell)

            
            #place predictions in a tensor holding predictions for each token
            outputs[:,t,:] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next inputs
            #if not, use predicted token
            inputs = trg[:,t,:] if teacher_force else output
        ## Store the last output
        #outputs[:,tq_len-1,:] = output
        return outputs



class EcoDblDco(nn.Module):
    """
    Sequence to sequence model with one encoder and two decoders.
    The encoder read the sequence data, and hidden unit of the encoder is the
    input to the two decoders. The one decoder repeat the input sequence from
    left to right, and the other decoder repeat the input sequence from right to left
    """
    def __init__(self, encoder, decoderL2R, decoderR2L, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoderL2R = decoderL2R
        self.decoderR2L = decoderR2L
        self.device = device
        #self.reverse = reverse
        
        assert encoder.hid_dim == decoderL2R.hid_dim ==decoderR2L.hid_dim , \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoderL2R.n_layers==decoderR2L.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trgL2R,trgR2L, teacher_forcing_ratio = 0.5):
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trgL2R.shape[0]
        ## Output length
        tq_len = trgL2R.shape[1]
        ## Output dimension (1)
        tq_dim = self.decoderL2R.output_dim
        
        #tensor to store decoder output left to right
        outputsL2R_seq = torch.zeros(batch_size,tq_len,tq_dim).to(self.device)
        #tensor to store decoder right to left
        outputsR2L_seq = torch.zeros(batch_size,tq_len,tq_dim).to(self.device)
        
        ## Get the hidden state of the encoder, 
        ## They will be used as the input to the two decoders
        src = src.to(self.device).float()
        if hasattr(self.decoderL2R, 'attention'): 
            encoder_outputs, hiddenL2R,hiddenR2L,cellL2R,cellR2L = self.encoder(src)
        else:
            encoder_outputsL2R, hiddenL2R, cellL2R = self.encoder(src)
            ## The two decoders share the same encoder, so clone the hidden and cell
            ## The gradient will be accumulated into corresponding leaf variables
            hiddenR2L = hiddenL2R.clone()
            cellR2L = cellL2R.clone()

        ## Initial input is always the 'gruond truth time and angle'
        #inputsL2R = copy.deepcopy(trgL2R[:,0,:])
        #inputsR2L = copy.deepcopy(trgR2L[:,0,:])
        ## Initial input is 0
        inputsL2R = torch.zeros(trgL2R[:,0,:].shape).to(self.device)
        inputsR2L = torch.zeros(trgL2R[:,0,:].shape).to(self.device)

        # The fist output is the 'ground truth value'
        #outputsL2R_seq[:,0,:] = inputsL2R
        #outputsR2L_seq[:,0,:] = inputsR2L
        
        for t in range(0, tq_len):
            ## If use attention model and double decoder. The two decoder share attention
            if hasattr(self.decoderL2R, 'attention'): 
                outputL2R, hiddenL2R, cellL2R = self.decoderL2R(inputsL2R.to(self.device).float(), hiddenL2R, cellL2R,encoder_outputs)
                hiddenL2R = hiddenL2R.permute(1,0,2)
                outputR2L, hiddenR2L, cellR2L = self.decoderR2L(inputsR2L.to(self.device).float(), hiddenR2L, cellR2L,encoder_outputs)
                hiddenR2L = hiddenR2L.permute(1,0,2)
            else:
                outputL2R, hiddenL2R, cellL2R = self.decoderL2R(inputsL2R.to(self.device).float(), hiddenL2R, cellL2R)
                outputR2L, hiddenR2L, cellR2L = self.decoderR2L(inputsR2L.to(self.device).float(), hiddenR2L, cellR2L)
            
            #place predictions in a tensor holding predictions for each token
            outputsL2R_seq[:,t,:] = outputL2R
            outputsR2L_seq[:,t,:] = outputR2L
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next inputsL2R
            #if not, use predicted token
            inputsL2R = trgL2R[:,t,:] if teacher_force else outputL2R
            inputsR2L = trgR2L[:,t,:] if teacher_force else outputR2L
        ## Store the last output
        #outputsL2R_seq[:,tq_len-1,:] = outputL2R
        #outputsR2L_seq[:,tq_len-1,:] = outputR2L
        return outputsL2R_seq, outputsR2L_seq
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        


def train(model, train_loader, optimizer, criterion, clip,teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    running_loss = 0
    for i, data in enumerate(train_loader):
        src, trg = data
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        
        #output_dim = output.shape[-1]
        #output = output.view(-1, output_dim)
        #trg = trg.view(-1, output_dim)
        
        loss = criterion(output, trg.to(model.device).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        

        ## Print statistics
        running_loss += loss.item()
        if i % 20 == 19: # Print every 20 mini-batches
            print('Running loss: %.6f' % (running_loss / 20))
            running_loss = 0.0

    
    return epoch_loss / len(train_loader)

def train_double_decoder(model, train_loader, optimizer, criterionL2R,criterionR2L, clip,teacher_forcing_ratio):
    model.train()
    epoch_lossL2R = 0
    running_lossL2R = 0
    epoch_lossR2L = 0
    running_lossR2L = 0
    
    for i, data in enumerate(train_loader):
        src, trg = data
        
        ## The input sequence, src, will not be reversed, it is the same with L2R,
        ## The target sequence, trg, is reversed.
        trgL2R = copy.deepcopy(src)
        trgR2L = copy.deepcopy(trg)
        
        optimizer.zero_grad()
        
        outputsL2R,outputsR2L = model(src, trgL2R,trgR2L, teacher_forcing_ratio)
        
        #output_dim = output.shape[-1]
        #output = output.view(-1, output_dim)
        #trg = trg.view(-1, output_dim)
        
        lossL2R = criterionL2R(outputsL2R, trgL2R.to(model.device).float())
        lossL2R.backward(retain_graph=True)
        #lossL2R.backward()
        
        lossR2L = criterionR2L(outputsR2L, trgR2L.to(model.device).float())
        lossR2L.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_lossL2R += lossL2R.item()
        epoch_lossR2L += lossR2L.item()
        ## Print statistics
        running_lossL2R += lossL2R.item()
        running_lossR2L += lossR2L.item()
        
        if i % 20 == 19: # Print every 20 mini-batches
            print('Running loss---- L2R: %.6f,  R2L: %.6f' % (running_lossL2R / 20,running_lossR2L / 20))
            running_lossL2R = 0.0
            running_lossR2L = 0.0

    
    return epoch_lossL2R / len(train_loader),epoch_lossR2L / len(train_loader)

def evaluate(model, valid_loader, criterion,seq_len, teacher_forcing_ratio):
    
    model.eval()
    
    ## This array stores ground truth and predicted value.
    #pair_array = np.zeros((len(valid_loader),2,seq_len,2))
    error_list = []
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            src, trg = data
            ## Different from translation, ground truth is always known, 
            ## so we can test teacher forcing rate
            output = model(src, trg, teacher_forcing_ratio) 
            output_cpu = output.cpu().data.numpy()
            figname_trace = fig_path + 'trac'

            
            figname_error = fig_path+"predtionerror"
            for i in range(len(output_cpu[0][1])):
                plot_pred_true(output_cpu[0,:,i],trg[0,:,i].data.numpy(),figname_trace+str(i)+'.png')
                error = output_cpu[0,:,i] - trg[0,:,i].data.numpy()
                plot_error(error,figname_error+str(i)+'.png')
                
            ## Plot norm (all dimensions) of error
            error_2d = output_cpu[0,:] - trg[0,:].data.numpy()
            norm = np.linalg.norm(error_2d, axis=1)
            plot_error(norm,figname_error+'_norm.png',color='y')
            
            error_list.append(norm)
        loss = criterion(output, trg.to(model.device).float())
        print("Prediction Error", loss.item())
        return error_list

def getSmallerErrors(outputL2R, outputR2L_rev, truth):
    """
    errL2R = np.abs(outputL2R - truth)
    errR2L = np.abs(outputR2L_rev - truth)
    minErr = np.minimum(errL2R,errR2L)
    """
    errL2R = outputL2R - truth
    if errL2R.shape[1]>1:
        errL2R = np.linalg.norm(errL2R, axis=1)
        
    errR2L = outputR2L_rev - truth
    if errR2L.shape[1]>1:
        errR2L = np.linalg.norm(errR2L, axis=1)
    minErr = np.zeros(len(truth))
    for i, (le, re) in enumerate(zip(errL2R, errR2L)):
        if abs(le)<abs(re):
            minErr[i] = le
        else:
            minErr[i] = re


    plt.figure(figsize=(figure_width,figure_hight))
    plt.plot(minErr, "-x", color="y", label="Error",markersize=4)
    plt.xticks(fontsize=legend_fontsize)
    plt.yticks(fontsize=legend_fontsize) #rotation=90
    #plt.ylim(-10,15)
    #plt.ylabel('Min Error')
    #plt.legend()
    #plt.grid(True)
    plt.savefig(fig_path+"error.png",dpi=400)
    
    return minErr
    
    
def evaluateEcoDblDco(model, valid_loader, criterionL2R,criterionR2L,seq_len, teacher_forcing_ratio):
    
    model.eval()
    
    ## This array stores ground truth and predicted value.
    #pair_array = np.zeros((len(valid_loader),2,seq_len,2))
    #error_list = {}
    min_error_list = []
    
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            src, trg = data
            ## The input sequence is not reversed, the left2Right prediction is the same with input sequence
            trgL2R = copy.deepcopy(src)
            ## The trg sequence is reversed, the Right2Left prediction is the same with the reversed trg
            trgR2L = copy.deepcopy(trg)
            ## Different from translation, ground truth is always known, 
            ## so we can test teacher forcing rate
            outputL2R, outputR2L = model(src, trgL2R,trgR2L, teacher_forcing_ratio) 
            output_cpuL2R = outputL2R.cpu().data.numpy()
            figname_trace = fig_path + 'traceL2R'
            figname_error = fig_path+"errorL2R"
            for i in range(output_cpuL2R.shape[2]):
                plot_pred_true(output_cpuL2R[0,:,i],trgL2R.data.numpy()[0,:,i],figname_trace+str(i)+'.png')
                error = output_cpuL2R[0,:,i] - trgL2R[0,:,i].data.numpy()
                #error_list['L2R'] = error
                #error_norm = np.linalg.norm(error, axis=0)
                plot_error(error,figname_error+str(i)+'.png')


            figname_trace = fig_path + 'traceR2L'
            figname_error = fig_path+"errorR2L"
            output_cpuR2L_rev = np.flip(outputR2L.cpu().data.numpy(),1)
            for i in range(output_cpuR2L_rev.shape[2]):
                plot_pred_true(outputR2L.cpu().data.numpy()[0,:,i],trgR2L.data.numpy()[0,:,i],figname_trace+str(i)+'.png')
                ## Reverse the reversed output. Make it left to right
                error = output_cpuR2L_rev[0,:,i] - trgL2R[0,:,i].data.numpy()
                #error_norm = np.linalg.norm(error, axis=0)
                #error_list['R2L']=error
                plot_error(error,figname_error+str(i)+'.png')
            
            min_err = getSmallerErrors(output_cpuL2R[0], output_cpuR2L_rev[0], trgL2R[0].data.numpy())
            #error_list['min'] = min_err
            min_error_list.append(min_err)
            
        lossL2R = criterionL2R(outputL2R, trgL2R.to(model.device).float())
        lossR2L = criterionR2L(outputR2L, trgR2L.to(model.device).float())

        print("Prediction Error:  L2R: %.3f, R2L: %.3f", (lossL2R.item(),lossR2L.item()))
        
        return min_error_list

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def run_train(model, train_loader, optimizer, criterion,n_epoch, clip,tr):
    for epoch in range(n_epoch):
        print("Eopch:", epoch+1)
        
        #start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer,criterion, clip,tr)
        #valid_loss = evaluate(model, valid_loader, criterion,TEACHER_FORCING_RATIO_PRED)
        
        #end_time = time.time()
        
        #epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print("Loss:%f" %(train_loss))



def run_trainEcoDblDco(model, train_loader, optimizer, criterionL2R,criterionR2L,n_epoch, clip,tr):
    for epoch in range(n_epoch):
        print("Eopch:", epoch+1)
        
        #start_time = time.time()
        
        train_lossL2R, train_lossR2L = train_double_decoder(model, train_loader, optimizer,criterionL2R,criterionR2L, clip,tr)
        #valid_loss = evaluate(model, valid_loader, criterion,TEACHER_FORCING_RATIO_PRED)
        
        #end_time = time.time()
        
        #epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print("Loss L2R:%f, R2L:%f" %(train_lossL2R,train_lossR2L))
        """
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        """
def readSavedModel(seq2seq_model,encoder_file,decoder_file,read_enc,read_dec):
    if read_enc == True:
        seq2seq_model.encoder.load_state_dict(torch.load(encoder_file))
    if read_dec == True:
        seq2seq_model.decoder.load_state_dict(torch.load(decoder_file))
        
def readSavedModelEcoDblDco(seq2seq_model,encoder_file,decoder_fileL2R,decoder_fileR2L,read_enc,read_dec):
    if read_enc == True:
        seq2seq_model.encoder.load_state_dict(torch.load(encoder_file))
    if read_dec == True:
        seq2seq_model.decoderL2R.load_state_dict(torch.load(decoder_fileL2R))
        seq2seq_model.decoderR2L.load_state_dict(torch.load(decoder_fileR2L))

def loadData(data_dir, num_data, rev_in=False,rev_trg=False, batch_size=4,
             load_saved_data=False,shuffle=False, smooth=False,readAngle=False,preprocess=False):
    addNum = 40
    if load_saved_data == False:
        if smooth==True:
            ## If data needs smoothing, add a few data at the boundary
            ## These data will be deleted, this avoid smoothing error.
            num_data = num_data + addNum*np.ones_like(num_data) 
        ## Read from Excel file (raw data), this takes longer to load and process the data
        engines = EngineData(data_dir)
        if readAngle==False:
            engines.readExcel()
        else:
            engines.readExcel([0,2,3])
        #engines.writeToCSV()
        engines.getSequence(num_data,readAngle)
        
        ## Scale the feature value to be between 0 and 1
        if preprocess:
            engines.preprocess()
            
        if smooth==True:
            engines.sequence = smoothTorque(engines.sequence,PADDLEN,TRUNCLEN,addNum)
            ## Save the sequence data to local, and can be read faster for future use
            np.save(data_dir+'sequence_smooth.npy',engines.sequence)
        else:
            ## Save the sequence data to local, and can be read faster for future use
            np.save(data_dir+'sequence.npy',engines.sequence)

    else:
        engines = EngineData('')
        if smooth == True:
            engines.sequence = np.load(data_dir+'sequence_smooth.npy')
        else:
            engines.sequence = np.load(data_dir+'sequence.npy')
    #plot_torque(time_angle,torques)
    print("Number of engines %d" %(engines.sequence.shape[0]))
    ## Training Data
    ## Dimension is [batch size, lenth of sequence, dimension]
    ## If rev_in is True, reverse the input sequence
    if rev_in == True:
        sequence_rev = copy.deepcopy(np.flip(engines.sequence,1))
        sequence_tensor_src = torch.from_numpy(sequence_rev)
    else:
        sequence = copy.deepcopy(engines.sequence)
        sequence_tensor_src = torch.from_numpy(sequence)
    ## If rev_trg is True, reverse the target sequence
    if rev_trg == True:
        sequence_rev = copy.deepcopy(np.flip(engines.sequence,1))
        sequence_tensor_trg = torch.from_numpy(sequence_rev)
    else:
        sequence = copy.deepcopy(engines.sequence)
        sequence_tensor_trg = torch.from_numpy(engines.sequence)
        
    ## Change dimension to [lenth of sequence,batch size, dimension]
    #sequence_tensor = sequence_tensor.permute(1,0,2) 
    data = TensorDataset(sequence_tensor_src,sequence_tensor_trg)
    data_loader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return data_loader

def smoothTorque(sequence,paddingLen,truncLen,addNum):
    zeros = np.zeros((paddingLen,1))
    sequence_smooth = np.zeros((len(sequence),sequence.shape[1]-addNum,1))
    for i,f in enumerate(sequence):
        f = np.concatenate([f,zeros])
        n = len(f)
        ## Fast Fourier transform
        fhat = np.fft.fft(f,n,axis=0)
        
        fhat[truncLen:(n-truncLen+1)] = 0
        ifft = np.fft.ifft(fhat,axis=0) # Inverse FFT for filtered time signal
        #f[:] = ifft.real        
        sequence_smooth[i] = ifft[0:sequence.shape[1]-addNum]
        """
        plt.figure()
        plt.plot(f[:240],color='c',LineWidth=1.5)
        plt.plot(sequence_smooth[i],color='r',LineWidth=1.5)
        """
    return sequence_smooth

def readModel(file_encoder,file_decoder,model):
    model.encoder.load_state_dict(torch.load(file_encoder))
    model.decoder.load_state_dict(torch.load(file_decoder))
    
def trainSeq2Seq(TRACE,paths,rev_in = True,rev_trg = False,read_enc = True,
                 read_dec = True,read_model=True,train=True,smooth=False,
                 readAngle=False,preprocess=False):
    """
    input: 
    model: Seq2Seq model
    rev_in: bool, if true, reverse the input sequence
    rev_trg: bool, if true, reverse the target sequence
    read_enc: bool, if true, read saved encoder
    read_dec: bool, if true, read saved decoder
    paths: dictionary, paths to files
    read_model: bool, if true, read saved model
    train: bool, if true, train the model
    readAngle: bool, if true, read angle
    preprocess: bool, if true, scale input between 0 and 1
    """
    encoder = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)

    outputNet = OutputNetwork(DEC_HID_DIM, OUTPUT_DIM)

    decoder = Decoder(outputNet, INPUT_DIM,HID_DIM, OUTPUT_DIM, N_LAYERS, DEC_DROPOUT)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    model.apply(init_weights)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    ## Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
        
    
    #best_valid_loss = float('inf')
    
    
    train_loader = loadData(paths['data_dir_train'], NUM_DATA, rev_in,rev_trg, BATCH,LOADSAVEDDATA_TRAIN, readAngle=readAngle,preprocess=preprocess )
    
    valid_loader = loadData(paths['data_dir_valid'], NUM_DATA, rev_in,rev_trg, BATCH_VALID,LOADSAVEDDATA_VALID, readAngle=readAngle,preprocess=preprocess)
    
    if read_model == True:
        readSavedModel(model,paths['model_path']+'encoder.pth',paths['model_path']+'decoder.pth',read_enc,read_dec)
    if train == True:
        run_train(model, train_loader, optimizer, criterion,N_EPOCHS, CLIP,TEACHER_FORCING_RATIO_TRAIN)
    
    torch.save(encoder.state_dict(),paths['model_path']+'encoder.pth')
    torch.save(decoder.state_dict(),paths['model_path']+'decoder.pth')
    
    error_list = evaluate(model, valid_loader, criterion, NUM_DATA, TEACHER_FORCING_RATIO_TRAIN)
    
    return model,error_list

def trainEcoDblDco(TRACE,paths,rev_in = True,rev_trg = False,read_enc = True,read_dec = True,read_model=True,train=True,smooth=False):
    """
    input: 
    model: Seq2Seq model
    rev_in: bool, if true, reverse the input sequence
    rev_trg: bool, if true, reverse the target sequence
    read_enc: bool, if true, read saved encoder
    read_dec: bool, if true, read saved decoder
    paths: dictionary, paths to files
    read_model: bool, if true, read saved model
    train: bool, if true, train the model
    """
    encoder = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)

    outputNet = OutputNetwork(DEC_HID_DIM, OUTPUT_DIM)
    
    decoderL2R = Decoder(outputNet, INPUT_DIM,HID_DIM, OUTPUT_DIM, N_LAYERS, DEC_DROPOUT)
    decoderR2L = Decoder(outputNet, INPUT_DIM,HID_DIM, OUTPUT_DIM, N_LAYERS, DEC_DROPOUT)
    
    model = EcoDblDco(encoder, decoderL2R, decoderR2L, device).to(device)
    
    model.apply(init_weights)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    ## Defining loss function and optimizer
    criterionL2R = nn.MSELoss()        
    criterionR2L = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    
    #best_valid_loss = float('inf')
    
    
    train_loader = loadData(paths['data_dir_train'], NUM_DATA, rev_in,rev_trg, BATCH,LOADSAVEDDATA_TRAIN,smooth=smooth)
    
    valid_loader = loadData(paths['data_dir_valid'], NUM_DATA, rev_in,rev_trg, BATCH_VALID,LOADSAVEDDATA_VALID,smooth=smooth)
    
    if read_model == True:
        readSavedModelEcoDblDco(model,paths['model_path']+'encoder.pth',paths['model_path']+'decoderL2R.pth',paths['model_path']+'decoderR2L.pth',read_enc,read_dec)
    if train == True:
        run_trainEcoDblDco(model, train_loader, optimizer, criterionL2R, criterionR2L,N_EPOCHS, CLIP,TEACHER_FORCING_RATIO_TRAIN)
    
        torch.save(encoder.state_dict(),paths['model_path']+'encoder_double.pth')
        torch.save(decoderL2R.state_dict(),paths['model_path']+'decoderL2R.pth')
        torch.save(decoderR2L.state_dict(),paths['model_path']+'decoderR2L.pth')
    
    error_list = evaluateEcoDblDco(model, valid_loader, criterionL2R, criterionR2L, NUM_DATA, TEACHER_FORCING_RATIO_PRED)
    
    return model,error_list

def trainSeq2SeqATT(TRACE,paths,rev_in=False ,rev_trg=False,read_model=True,train=True,smooth=False,readAngle=False,preprocess=False):
    """
    input: 
    model: Seq2SeqATT model
    rev_in: bool, if true, reverse the input sequence
    rev_trg: bool, if true, reverse the target sequence
    read_enc: bool, if true, read saved encoder
    read_dec: bool, if true, read saved decoder
    paths: dictionary, paths to files
    read_model: bool, if true, read saved model
    train: bool, if true, train the model
    """
    doubledecoder = False
    encoder = EncoderATT(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT,doubledecoder)
    attention = Attention(ENC_HID_DIM, DEC_HID_DIM)

    outputNet = OutputNetwork(DEC_HID_DIM, OUTPUT_DIM)
    decoder = DecoderATT(outputNet,attention,INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    model.apply(init_weights)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    ## Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
        
    
    #best_valid_loss = float('inf')
    
    
    train_loader = loadData(paths['data_dir_train'], NUM_DATA, rev_in,rev_trg, BATCH,LOADSAVEDDATA_TRAIN,readAngle=readAngle, preprocess=preprocess)
    
    valid_loader = loadData(paths['data_dir_valid'], NUM_DATA, rev_in,rev_trg, BATCH_VALID,LOADSAVEDDATA_VALID,readAngle=readAngle, preprocess=preprocess)
    
    if read_model == True:
        readSavedModel(model,paths['model_path']+'encoderATT.pth',paths['model_path']+'decoderATT.pth',read_enc=True,read_dec=True)
    if train == True:
        run_train(model, train_loader, optimizer, criterion,N_EPOCHS, CLIP,TEACHER_FORCING_RATIO_TRAIN)
        torch.save(encoder.state_dict(),paths['model_path']+'encoderATT.pth')
        torch.save(decoder.state_dict(),paths['model_path']+'decoderATT.pth')
    
    error_list = evaluate(model, valid_loader, criterion, NUM_DATA, TEACHER_FORCING_RATIO_TRAIN)
    
    return model, error_list

def trainSeq2SeqATT_DblDco(TRACE,paths,rev_in = True,rev_trg = False,read_enc = True,read_dec = True,read_model=True,train=True,shuffle=True,smooth=False,readAngle=False,preprocess=False):
    """
    Attention model with double decoder, the second decoder predict from right to left.
    The sturcture is the same with function trainSeq2SeqATT
    """
    doubledecoder = True
    encoder = EncoderATT(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT,doubledecoder)
    attention = Attention(ENC_HID_DIM, DEC_HID_DIM)
    
    outputNet = OutputNetwork(DEC_HID_DIM, OUTPUT_DIM)
    decoder_L2R = DecoderATT(outputNet,attention, INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT)
    #outputNet2 = OutputNetwork(DEC_HID_DIM, OUTPUT_DIM)
    decoder_R2L = DecoderATT(outputNet,attention, INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT)
    
    model = EcoDblDco(encoder, decoder_L2R,decoder_R2L, device).to(device)
    
    model.apply(init_weights)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    ## Defining loss function and optimizer
    criterion_L2R = nn.MSELoss()
    criterion_R2L = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
        
    
    #best_valid_loss = float('inf')
    
    
    train_loader = loadData(paths['data_dir_train'], NUM_DATA, rev_in,rev_trg, BATCH,LOADSAVEDDATA_TRAIN,shuffle, smooth,readAngle=readAngle,preprocess=preprocess)
    
    valid_loader = loadData(paths['data_dir_valid'], NUM_DATA, rev_in,rev_trg, BATCH_VALID,LOADSAVEDDATA_VALID,shuffle, smooth,readAngle=readAngle,preprocess=preprocess)
    
    if read_model == True:
        readSavedModelEcoDblDco(model,paths['model_path']+'encoderATT_double.pth',paths['model_path']+'decoderATTL2R.pth',paths['model_path']+'decoderATTR2L.pth',read_enc,read_dec)
    if train == True:
        run_trainEcoDblDco(model, train_loader, optimizer, criterion_L2R,criterion_R2L,N_EPOCHS, CLIP,TEACHER_FORCING_RATIO_TRAIN)
    
        torch.save(encoder.state_dict(),paths['model_path']+'encoderATT_double.pth')
        torch.save(decoder_L2R.state_dict(),paths['model_path']+'decoderATTL2R.pth')
        torch.save(decoder_R2L.state_dict(),paths['model_path']+'decoderATTR2L.pth')
    
    error_list = evaluateEcoDblDco(model, valid_loader, criterion_L2R,criterion_R2L, NUM_DATA, TEACHER_FORCING_RATIO_TRAIN)
    
    return model, error_list

def updateLearningRate(optimizer,learnrate,model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)

def nparraytoxlsx(nparray):
    df = pd.DataFrame(nparray)
    df.to_excel('min_error.xlsx',index=False)
    



## Valve Lash (VL) or Torque To Turn (TTT)
TRACE = 'VL'

if torch.cuda.is_available():
    device = torch.device('cuda')
    if TRACE == 'VL':
        data_dir_train = 'drive/My Drive/Colab Notebooks/Engine/data/VL/train/'
        data_dir_valid = 'drive/My Drive/Colab Notebooks/Engine/data/VL/valid/'
        NUM_DATA = [240,240,240]
        model_path = 'drive/My Drive/Colab Notebooks/Engine/SavedModels/VL/'
    else:
        data_dir_train = 'drive/My Drive/Colab Notebooks/Engine/data/TTT/train/'
        data_dir_valid = 'drive/My Drive/Colab Notebooks/Engine/data/TTT/valid/'
        NUM_DATA = [65,200,600]
        model_path = 'drive/My Drive/Colab Notebooks/Engine/SavedModels/TTT/'
    fig_path = 'drive/My Drive/Colab Notebooks/Engine/Fig/'
    N_EPOCHS = 6
    BATCH = 8
else:
    device = torch.device('cpu')
    ## Data is in one folder up. Not up loaded to github
    cwd_up = os.path.dirname(os.getcwd())
    
    if TRACE == 'VL':
        data_dir_train = cwd_up+"/data/test/VL/train/"
        data_dir_valid = cwd_up + "./data/test/VL/valid/"
        model_path = cwd_up + "./model/VL/"
        NUM_DATA = [240,240,240]
    else:
        data_dir_train = cwd_up+"/data/test/TTT/train/"
        data_dir_valid = cwd_up + "./data/test/TTT/valid/"
        model_path = cwd_up + "./model/TTT/"
        NUM_DATA = [65,200,600]
    fig_path = "./Fig/"
    N_EPOCHS = 1
    BATCH = 4
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
print('device:',device)



#torques = np.zeros((len(engines_train),NUM_DATA))
LEARN_RATE = 0.001/5
READANGLE = True
PREPROCESS = True
if READANGLE==True:
    INPUT_DIM = 2
    OUTPUT_DIM = 2
else:
    INPUT_DIM = 1
    OUTPUT_DIM = 1
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0
DEC_DROPOUT = 0
CLIP = 1
TEACHER_FORCING_RATIO_TRAIN = 0
TEACHER_FORCING_RATIO_PRED = 0
PADDLEN = 1000 ##Length in zeropadding for FFT
TRUNCLEN=100 ## Truncation lenth of frequence after FFT
BATCH_VALID = 1

LOADSAVEDDATA_TRAIN = True
LOADSAVEDDATA_VALID = True

rev_in = False
rev_trg = True

read_enc = True
read_dec = True
read_model = True

TRAIN = True
smooth = False
paths = {'data_dir_train':data_dir_train,
         'data_dir_valid':data_dir_valid,
         'model_path':model_path}
shuffle = True

ENC_HID_DIM=512
DEC_HID_DIM=512


model,error_list = trainSeq2Seq(TRACE, paths, rev_in, rev_trg,read_enc,read_dec,read_model,TRAIN,readAngle=READANGLE,preprocess=PREPROCESS)

## The input sequence, src, will not be reversed, it is the same with L2R,
## The target sequence, trg, is reversed.
#model,error_list = trainEcoDblDco(TRACE,paths,rev_in, rev_trg , read_enc ,read_dec , read_model,TRAIN,smooth)



model,error_list = trainSeq2SeqATT(TRACE,paths,rev_in ,rev_trg , read_model,TRAIN,smooth,readAngle=READANGLE,preprocess=PREPROCESS)

model,error_list = trainSeq2SeqATT_DblDco(TRACE,paths,rev_in, rev_trg , read_enc ,read_dec , read_model,TRAIN,shuffle,smooth,readAngle=READANGLE,preprocess=PREPROCESS)

#upper,lower = error_limit(error_list)