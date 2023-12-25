import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class ConvGRUCell(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 2 * self.hidden_channels
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=self.hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=self.dilation,
            bias=bias)

    def forward(self, input, hidden):
        hidden = hidden[0]
        gi = self.conv_ih(input)
        gh = self.conv_hh(hidden)
        i_r, i_u = gi.chunk(2, 1)
        h_r, h_u = gh.chunk(2, 1)

        updategate = torch.sigmoid(i_u + h_u)
        resetgate = torch.sigmoid(i_r + h_r)
        
        new_h = torch.tanh(i_r + resetgate * h_r)
        hy = new_h + updategate * (hidden - new_h)

        return hy, hy

        
        