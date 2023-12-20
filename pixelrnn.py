#https://github.com/heechan95/PixelRNN-pytorch/blob/master/PixelRNN%20pytorch.ipynb
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask='B', **kargs):
        super(MaskedConv2d, self).__init__(*args, **kargs)
        assert mask in {'A', 'B'}
        self.mask_type = mask
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
    
        _, _, H, W = self.mask.size()
    
        self.mask[:, :, H//2,W//2 + (self.mask_type == 'B'):] = 0
        self.mask[:, :, H//2+1:, :] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        #print(self.weight.data)
        return super(MaskedConv2d, self).forward(x)
    
class MaskedConv1d(nn.Conv1d):
    def __init__(self, *args, mask='B', **kargs):
        super(MaskedConv1d, self).__init__(*args, **kargs)
        assert mask in {'A', 'B'}
        self.mask_type = mask
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
    
        _, _, W = self.mask.size()
    
        self.mask[:, :, W//2 + (self.mask_type == 'B'):] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)

def _padding(i, o, k, s=1, d=1, mode='same'):
    if mode == 'same':
        return ((o-1) * s + (k-1)*(d-1) + k - i) // 2
    else:
        raise RuntimeError('Not implemented')
    
class RowLSTMCell(nn.Module):
    def __init__(self, hidden_dims, image_size, channel_in, *args, **kargs):
        super(RowLSTMCell, self).__init__(*args, **kargs)

        self._hidden_dims = hidden_dims
        self._image_size = image_size
        self._channel_in = channel_in
        self._num_units = self._hidden_dims * self._image_size
        self._output_size = self._num_units
        self._state_size = self._num_units * 2

        self.conv_i_s = MaskedConv1d(self._hidden_dims, 4 * self._hidden_dims, 3, mask='B', padding=_padding(image_size, image_size, 3))
        self.conv_s_s = nn.Conv1d(channel_in, 4 * self._hidden_dims, 3, padding=_padding(image_size, image_size, 3))
   
    def forward(self, inputs, states):
        c_prev, h_prev = states



        h_prev = h_prev.view(-1, self._hidden_dims,  self._image_size)
        inputs = inputs.view(-1, self._channel_in, self._image_size)

        s_s = self.conv_s_s(h_prev) #(batch, 4*hidden_dims, width)
        i_s = self.conv_i_s(inputs) #(batch, 4*hidden_dims, width)



        s_s = s_s.view(-1, 4 * self._num_units) #(batch, 4*hidden_dims*width)
        i_s = i_s.view(-1, 4 * self._num_units) #(batch, 4*hidden_dims*width)

        #print(s_s.size(), i_s.size())

        lstm = s_s + i_s

        lstm = torch.sigmoid(lstm)

        i, g, f, o = torch.split(lstm, (4 * self._num_units)//4, dim=1)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        new_state = (c, h)
        return h, new_state
    
    
class RowLSTM(nn.Module):
    def __init__(self, hidden_dims, input_size, channel_in, *args, init='zero', **kargs):
        super(RowLSTM, self).__init__(*args, **kargs)
        assert init in {'zero', 'noise', 'variable', 'variable noise'}

        self.init = init
        self._hidden_dims = hidden_dims
        #self.return_state = return_state
        if self.init == 'zero':
            self.init_state = (torch.zeros(1, input_size * hidden_dims), torch.zeros(1, input_size * hidden_dims))
        elif self.init == 'noise':
            self.init_state = (torch.Tensor(1, input_size * hidden_dims), torch.Tensor(1, input_size * hidden_dims))
            nn.init.uniform(self.init_state[0])
            nn.init.uniform(self.init_state[1])  
        elif self.init == 'variable':
            hidden0 = torch.zeros(1,input_size * hidden_dims)
            ##if use_cuda:
            ##  hidden0 = hidden0.cuda()
            ##else:
            ##  hidden0 = hidden0
            self._hidden_init_state = torch.nn.Parameter(hidden0, requires_grad=True)
            self._cell_init_state = torch.nn.Parameter(hidden0, requires_grad=True)
            self.init_state = (self._hidden_init_state, self._cell_init_state)
        else:
            hidden0 = torch.Tensor(1, input_size * hidden_dims) # size
            nn.init.uniform(hidden0)
            self._hidden_init_state = torch.nn.Parameter(hidden0, requires_grad=True)
            self._cell_init_state = torch.nn.Parameter(hidden0, requires_grad=True)
            self.init_state = (self._hidden_init_state, self._cell_init_state)

        self.lstm_cell = RowLSTMCell(hidden_dims, input_size, channel_in)
    
    def forward(self, inputs, initial_state=None):
        '''
        states --> (c, h), tuple
        c,h --> (batch, width * hidden_dims)
        inputs --> (batch, seq_length, input_shape)
        input_shape --> width, channel
        '''


        n_batch, channel, n_seq, width = inputs.size()
        #print(n_seq)
        #inputs = inputs.view(n_batch, channel, n_seq, width)
        if initial_state is None:
            hidden_init, cell_init = self.init_state
            hidden_init, cell_init = hidden_init.to(inputs.device), cell_init.to(inputs.device)

        else:
            hidden_init, cell_init = initial_state

        states = (hidden_init.repeat(n_batch,1), cell_init.repeat(n_batch, 1))

        steps = [] # --> (batch, width * hidden_dims) --> (batch, 1, width*hidden_dims)
        for seq in range(n_seq):
            #print(inputs[:, :, seq, :].size())
            h, states = self.lstm_cell(inputs[:, :, seq, :], states)
            steps.append(h.unsqueeze(1))

        return torch.cat(steps, dim=1).view(-1, n_seq, width, self._hidden_dims).permute(0,3,1,2) # --> (batch, seq_length(a.k.a height), width * hidden_dims)

class PixelRNN(nn.Module):
    def __init__(self, num_layers, hidden_dims, input_size, *args, **kargs):
        super(PixelRNN, self).__init__(*args, **kargs)

        pad_conv1 = _padding(input_size, input_size, 7)
        pad_conv2 = _padding(input_size, input_size, 1)
        self.conv1 = MaskedConv2d(32, hidden_dims, (7,7), mask='A', padding=(pad_conv1, pad_conv1))
        self.lstm_list = nn.ModuleList([RowLSTM(hidden_dims, input_size, hidden_dims) for _ in range(num_layers)])
        self.conv2 = nn.Conv2d(hidden_dims, 32, (1,1), padding=(pad_conv2, pad_conv2))
        self.conv_last = nn.Conv2d(32, 32, (1,1), padding=(pad_conv2, pad_conv2))
    
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        for lstm in self.lstm_list:
            x = lstm(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv_last(x)
        return torch.sigmoid(x)

if __name__ == '__main__':
    model = PixelRNN(num_layers=2, hidden_dims=16, input_size=64)
    x = torch.randn(16,32,64,64) 
    output = model(x)
    print("output:",output.shape)
