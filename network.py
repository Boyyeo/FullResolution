import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell, Sign


class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        self.conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, input, hidden1, hidden2, hidden3):
        x = self.conv(input)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]

        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = F.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self):
        super(DecoderCell, self).__init__()

        self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rnn1 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.rnn4 = ConvLSTMCell(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.conv2 = nn.Conv2d(
            32, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, hidden4):
        x = self.conv1(input)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

        hidden4 = self.rnn4(x, hidden4)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)

        x = F.tanh(self.conv2(x)) / 2
        return x, hidden1, hidden2, hidden3, hidden4


class GainEstimatorCell(nn.Module):
    def __init__(self):
        super(GainEstimatorCell, self).__init__()

        # [32, 3, 32, 32] floor((w+2*pad-ks)/s)+1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1) # [32, 32, 16, 16]

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1) # [32, 32, 8, 8]

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1) # [32, 32, 4, 4]

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1) # [32, 32, 2, 2]

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=2, stride=2, padding=1) # [32, 1, 2, 2]

    def forward(self, x):
        y1 = self.conv1(x)
        y1_ = F.elu(y1)
        y2 = self.conv2(y1_)
        y2_ = F.elu(y2)
        y3 = self.conv3(y2_)
        y3_ = F.elu(y3)
        y4 = self.conv4(y3_)
        y4_ = F.elu(y4)
        y5 = self.conv5(y4_)
        y5_ = F.elu(y5)

        # Add a constant of 2 to the output
        output = y5_ + 2.0

        return output