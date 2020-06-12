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



class EngineData:
    def __init__(self,path):
        self.path = path
        self.num_engines = -1
        
    def readExcel(self):
        df_array = []
        for file in tqdm_notebook(os.listdir(self.path)): 
            ## Ignore other file types 
            if file[-5:] != ".xlsx" or file[0:2] == '~$':
                continue

            ## Store csv file in a Pandas DataFrame. Only read 'Id' and 'Torque'
            df = pd.read_excel(self.path + file,usecols=[0,2])
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

    def getSequence(self,seq_len):
        if self.num_engines<0:
            self.getNumOfEngines()
        self.sequence = np.zeros((self.num_engines,seq_len,1))
        
        ## Set values
        index = -1
        for i,row in self.rawdata.iterrows():
            id = int(row['Id'])
            if id<seq_len:
                if id==0:
                    index+=1
                self.sequence[index][id] = row['Torque']


def plot_torque(angle,torques):
    plt.figure(figsize=(14,10))
    plt.plot(angle,torques, color="g")
    plt.ylabel('Torque')
    plt.legend()


def plot_pred_true(pred,truth):
    plt.figure()
    plt.plot(pred, "-x", color="g", label="Predicted",markersize=4)
    plt.plot(truth, color="b", label="Actual")
    plt.ylabel('Angle-Torque')
    plt.legend()
    plt.savefig(fig_path+"pred_true.png",dpi=400)

def plot_error(pred,truth):
    plt.figure()
    error = np.abs(pred-truth)
    plt.plot(error, "-x", color="g", label="Error",markersize=4)
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(fig_path+"error.png",dpi=400)
    
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
        
        return hidden, cell
    """
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hid_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hid_dim).zero_().to(device))
        return hidden
    """

class Decoder(nn.Module):
    def __init__(self,input_dim, hid_dim,output_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.hid_dim2 = int(hid_dim/2)
        self.hid_dim3 = int(self.hid_dim2/2)
        self.n_layers = n_layers
        self.input_dim = input_dim
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers,batch_first=False, dropout = dropout)

        self.fc1 = nn.Linear(self.hid_dim, self.hid_dim2)
        self.fc2 = nn.Linear( self.hid_dim2, self.hid_dim3)
        self.fc3 = nn.Linear(self.hid_dim3, self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, hidden, cell):
        
        inputs = inputs.unsqueeze(0)
        #inputs = self.dropout(inputs)
        output, (hidden, cell) = self.rnn(inputs, (hidden, cell)) 
        output = self.relu(output.squeeze(0))## Is the relu necessary?
        output = self.relu(self.fc1(output))
        output = self.relu(self.fc2(output))## For the top layer last step, output==hidden
        prediction = self.fc3(output)
        ## Try adding a second linear layer

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
        hidden, cell = self.encoder(src)
        
        ## Initial input is always the 'gruond truth time and angle'
        inputs = copy.deepcopy(trg[:,0,:])

        for t in range(0, tq_len-1):
            output, hidden, cell = self.decoder(inputs.to(self.device).float(), hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:,t,:] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next inputs
            #if not, use predicted token
            inputs = trg[:,t,:] if teacher_force else output
        ## Store the last output
        outputs[:,tq_len-1,:] = output
        return outputs

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
            print('Running loss: %.3f' % (running_loss / 20))
            running_loss = 0.0

    
    return epoch_loss / len(train_loader)


def evaluate(model, valid_loader, criterion,seq_len, teacher_forcing_ratio):
    
    model.eval()
    epoch_loss = 0
    
    ## This array stores ground truth and predicted value.
    #pair_array = np.zeros((len(valid_loader),2,seq_len,2))
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            src, trg = data
            ## Different from translation, ground truth is always known, 
            ## so we can test teacher forcing rate
            output = model(src, trg, teacher_forcing_ratio) 
            output_cpu = output.cpu().data.numpy()
            plot_pred_true(output_cpu[0],trg[0].data.numpy())
            plot_error(output_cpu[0],trg[0].data.numpy())
        loss = criterion(output, trg.to(model.device).float())
        print("Prediction Error", loss.item())


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
        """
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        """
        
def readSavedModel(seq2seq_model,encoder_file,decoder_file):
    seq2seq_model.encoder.load_state_dict(torch.load(encoder_file))
    seq2seq_model.decoder.load_state_dict(torch.load(decoder_file))
    
def loadData(data_dir, reverse=False, num_data=261,batch_size=4,load_saved_data=False,shuffle=False):
    if load_saved_data == False:
        ## Read from Excel file (raw data), this takes longer to load and process the data
        engines = EngineData(data_dir)
        engines.readExcel()
        #engines.writeToCSV()
        engines.getSequence(num_data)
        ## Save the sequence data to local, and can be read faster for future use
        np.save(data_dir+'sequence.npy',engines.sequence)

    else:
        engines = EngineData('')
        engines.sequence = np.load(data_dir+'sequence.npy')
    #plot_torque(time_angle,torques)
    print("Number of engines %d" %(engines.sequence.shape[0]))
    ## Training Data
    ## Dimension is [batch size, lenth of sequence, dimension]
    ## Reverse the source sequence but not the target sequence
    if reverse == True:
        sequence_rev = copy.deepcopy(np.flip(engines.sequence,1))
        sequence_tensor_src = torch.from_numpy(sequence_rev)
    else:
        sequence = copy.deepcopy(engines.sequence)
        sequence_tensor_src = torch.from_numpy(sequence)
    sequence_tensor_trg = torch.from_numpy(engines.sequence)
    ## Change dimension to [lenth of sequence,batch size, dimension]
    #sequence_tensor = sequence_tensor.permute(1,0,2) 
    data = TensorDataset(sequence_tensor_src,sequence_tensor_trg)
    data_loader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return data_loader


def readModel(file_encoder,file_decoder,model):
    model.encoder.load_state_dict(torch.load(file_encoder))
    model.decoder.load_state_dict(torch.load(file_decoder))
    
    

## Valve Lash (VL) or Torque To Turn (TTT)
TRACE = 'VL'

if torch.cuda.is_available():
    device = torch.device('cuda')
    if TRACE == 'VL':
        data_dir_train = 'drive/My Drive/Colab Notebooks/Engine/data/VL/train/'
        data_dir_valid = 'drive/My Drive/Colab Notebooks/Engine/data/VL/valid/'
        NUM_DATA = 240
    else:
        data_dir_train = 'drive/My Drive/Colab Notebooks/Engine/data/TTT/train/'
        data_dir_valid = 'drive/My Drive/Colab Notebooks/Engine/data/TTT/valid/'
        NUM_DATA = 600
    model_path = 'drive/My Drive/Colab Notebooks/Engine/'
    fig_path = 'drive/My Drive/Colab Notebooks/Engine/Fig/'
    N_EPOCHS = 20
    BATCH = 64
else:
    device = torch.device('cpu')
    ## Data is in one folder up. Not up loaded to github
    cwd_up = os.path.dirname(os.getcwd())
    model_path = cwd_up + "./model/"
    if TRACE == 'VL':
        data_dir_train = cwd_up+"/data/test/VL/train/"
        data_dir_valid = cwd_up + "./data/test/VL/valid/"
        NUM_DATA = 240
    else:
        data_dir_train = cwd_up+"/data/test/TTT/train/"
        data_dir_valid = cwd_up + "./data/test/TTT/valid/"
        NUM_DATA = 600
    fig_path = "./Fig/"
    N_EPOCHS = 1
    BATCH = 4
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
print('device:',device)



#torques = np.zeros((len(engines_train),NUM_DATA))
LEARN_RATE = 0.001

INPUT_DIM = 1
OUTPUT_DIM = 1
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0
DEC_DROPOUT = 0
CLIP = 1
TEACHER_FORCING_RATIO_TRAIN = 0
TEACHER_FORCING_RATIO_PRED = 0
#BATCH = 64
REVERSE = True
LOADSAVEDDATA = True


encoder = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(INPUT_DIM,HID_DIM, OUTPUT_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(encoder, decoder, device).to(device)

model.apply(init_weights)

print(f'The model has {count_parameters(model):,} trainable parameters')

## Defining loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    
criterion = nn.MSELoss()


best_valid_loss = float('inf')


train_loader = loadData(data_dir_train, REVERSE, NUM_DATA,BATCH,LOADSAVEDDATA)

valid_loader = loadData(data_dir_valid, REVERSE, NUM_DATA,BATCH,LOADSAVEDDATA)


run_train(model, train_loader, optimizer, criterion,N_EPOCHS, CLIP,TEACHER_FORCING_RATIO_TRAIN)
torch.save(encoder.state_dict(),model_path+'encoder.pth')
torch.save(decoder.state_dict(),model_path+'decoder.pth')

#readSavedModel(model,model_path+'encoder.pth',model_path+'decoder.pth')

evaluate(model, valid_loader, criterion, NUM_DATA, TEACHER_FORCING_RATIO_TRAIN)

