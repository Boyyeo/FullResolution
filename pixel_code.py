import unidecode
import string
import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
from pixelrnn import PixelRNN
from tqdm import tqdm 
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#Generate pseudorandom data
def generate_data(num_samples,p1,markovity):
    p0 = 1 - p1
    data = np.empty([num_samples,1],dtype=np.uint8)
    print(data.shape)
    data[:markovity] = np.random.choice([0, 1], size=(markovity,1), p=[p0, p1])
    for i in tqdm(range(markovity, num_samples)):
        if data[i-1] == data[i-markovity]:
            data[i] = 0
        else:
            data[i] = 1
    return data

#reads in a file using unicode symbols (eg: a text file)
def read_unicode_file(file_name):
    data = open(file_name).read()
    return data

#outputs a training batch
def get_training_batch(batch,n_batches,data):
    data_len = len(data)
    chunk_size = int(data_len/n_batches)
    chunk_start = batch*chunk_size
    chunk = data[chunk_start:(batch+1)*chunk_size]
    inp = char_to_tensor(chunk[:])
    target = char_to_tensor(chunk[:])
    return inp, target

# Turn string into a list of longs
def char_to_tensor(inp):
    all_characters = string.printable
    tensor = torch.zeros(len(inp)).long()
    for c in range(len(inp)):
        tensor[c] = all_characters.index(inp[c])
    return Variable(tensor)

def train_step(inp, target, model, criterion, optimizer):
    model.zero_grad()
    '''
    inp -> [bs*32*H*W]
    '''
    inp = inp.reshape(batch_size,32,H,W)

    output = model(inp) #[batch_size,32,H,W]
    inp = inp.flatten(start_dim=1) #[batch_size,32*H*W]
    output = output.flatten(start_dim=1) #[batch_size,32*H*W]
    loss = -(inp * torch.log(output) + (1 - inp) * torch.log(1 - output)).mean()
    loss.backward()
    optimizer.step()
    return loss.data.item()

def train_model(data):
    
    rnn_model = PixelRNN(num_layers=2, hidden_dims=16, input_size=64).to(device)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)
    loss_criterion = nn.CrossEntropyLoss()

    start = time.time()
    all_losses = []
    loss_avg = 0
    hidden=None

    for epoch in range(n_epochs):
        for batch in tqdm(range(iteration_per_epoch)):
            inp,target = get_training_batch(batch,iteration_per_epoch,data)
            inp = inp.to(device=device,dtype=torch.float32)


            #print("inp:{} target:{}".format(inp.shape,target.shape))
            loss = train_step(inp,target,rnn_model,loss_criterion,rnn_optimizer)       
            loss_avg += loss

            if batch % print_every == 0:
                print('[%s (%d, %d, %d%%) %.4f]' % (time_since(start), epoch, batch, (batch+ epoch*iteration_per_epoch)/(n_epochs*iteration_per_epoch)* 100, loss))
                #print(evaluate('Wh', 100), '\n')
                print('avg_loss: \n', loss_avg / print_every)
                all_losses.append(loss_avg / print_every)
                loss_avg = 0


def main():
    file_name="data/input.txt"
    data = generate_data(n_samples,0.5,256)
    #print("data:",data)
    np.savetxt(file_name,data,delimiter='', fmt='%d',newline='')
    data = read_unicode_file(file_name)
    #print("data:",data)
    train_model(data)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_epochs = 5
    iteration_per_epoch = 100
    batch_size = 4
    H,W,Channel = 64,64,32
    n_samples = iteration_per_epoch * batch_size * H * W * Channel
    print_every = 100
    hidden_size = 64
    n_layers = 1
    lr = 0.0005
    n_characters = 2
    main()

