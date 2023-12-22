import argparse
import dataclasses
import itertools
import warnings

import os
import time

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import torchvision
import matplotlib.pyplot as plt
import matplotlib
from pixelrnn import PixelRNN 
from entropy import * 
from arithmetic_compressor.rnn_compress import AECompressor_RNN
from arithmetic_compressor.util import Range
try:
  import torchac
except ImportError:
  raise ImportError('torchac is not available! Please see the main README '
                    'on how to install it.')


# Set to true to write to disk.
_WRITE_BITS = False


# Interactive plots setup
_FORCE_NO_INTERACTIVE_PLOTS = int(os.environ.get('NO_INTERACTIVE', 0)) == 1


if not _FORCE_NO_INTERACTIVE_PLOTS:
  try:
    matplotlib.use("TkAgg")
    interactive_plots_available = True
  except ImportError:
    warnings.warn(f'*** TkAgg not available! Saving plots...')
    interactive_plots_available = False
else:
  interactive_plots_available = False

if not interactive_plots_available:
  matplotlib.use("Agg")


# Set seed.
torch.manual_seed(0)


def train_test_loop(use_gpu=True,
                    bottleneck_size=32,
                    L=3,
                    batch_size=32,
                    lr=1e-4,
                    rate_loss_enable_itr=500,
                    num_test_batches=10,
                    train_plot_every_itr=50,
                    max_training_itr=None,
                    mnist_download_dir='data',
                    ):
  """Train and test an autoencoder.

  :param use_gpu: Whether to use the GPU, if it is available.
  :param bottleneck_size: Number of channels in the bottleneck.
  :param L: Number of levels that we quantize to.
  :param batch_size: Batch size we train with.
  :param lr: Learning rate of Adam.
  :param rate_loss_enable_itr: Iteration when the rate loss is enabled.
  :param num_test_batches: Number of batches we test on (randomly chosen).
  :param train_plot_every_itr: How often to update the train plot.
  :param max_training_itr: If given, only train for max_training_itr iterations.
  :param mnist_download_dir: Where to store MNIST.
  """
  ae = Autoencoder(bottleneck_size, L)
  #prob = ConditionalProbabilityModel(L=L, bottleneck_shape=ae.bottleneck_shape)
  prob = RNNConditionalProbabilityModel()

  device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
  ae = ae.to(device)
  prob = prob.to(device)

  mse = nn.MSELoss()
  adam = torch.optim.Adam(
    itertools.chain(ae.parameters(), prob.parameters()),
    lr=lr)

  train_acc = Accumulator()
  test_acc = Accumulator()
  plotter = Plotter()

  transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(
      # Make images 32x32.
      lambda image: F.pad(image, pad=(2, 2, 2, 2), mode='constant'))
  ])

  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(mnist_download_dir, train=True, download=True,
                               transform=transforms),
    batch_size=batch_size, shuffle=True)

  rate_loss_enabled = False
  total_epoch = 50
  for epoch in range(total_epoch):
    for i, (images, labels) in enumerate(train_loader):
        if max_training_itr and i >= max_training_itr:
            break
        assert images.shape[-2:] == (32, 32)
        images = images.to(device)
        labels = labels.to(device)

        adam.zero_grad()

        # Get reconstructions and symbols from the autoencoder.
        reconstructions, sym = ae(images)
        assert sym.shape[1:] == ae.bottleneck_shape

        # Get estimated and real bitrate from probability model, given labels.
        bits_estimated, bits_real = prob(sym.detach())
        mse_loss = mse(reconstructions, images)

        # If we are beyond iteration `rate_loss_enable_itr`, enable a rate loss.
        if i < rate_loss_enable_itr:
            loss = mse_loss
        else:
            loss = mse_loss + 1/1000 * bits_estimated
        rate_loss_enabled = True

        loss.backward()
        adam.step()

        # Update Train Plot.
        if i > 0 and i % train_plot_every_itr == 0:
            train_acc.append(i, bits_estimated, bits_real, mse_loss)
            print(f'{i: 10d}; '
                    f'loss={loss:.3f}, '
                    f'bits_estimated={bits_estimated:.3f}, '
                    f'bits_real={bits_real:.3f}, '
                    f'mse={mse_loss:.3f}')
            plotter.update('Train',
                            images, reconstructions, sym, train_acc, rate_loss_enabled)

        # Update Test Plot.
        if i > 0 and i % 100 == 0:
            ##print(f'{i: 10d} Testing on {num_test_batches} random batches...')
            ae.eval()
            prob.eval()
            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                mnist_download_dir, train=False, download=True, transform=transforms),
                batch_size=batch_size, shuffle=True)
            with torch.no_grad():
                across_batch_acc = Accumulator()
                for j, (test_images, test_labels) in enumerate(test_loader):
                    if j >= num_test_batches:
                        break
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)
                    test_reconstructions, test_sym = ae(test_images)
                    test_bits_estimated, test_bits_real = prob(test_sym)
                    #print("test_sym:{} length:{}".format(test_sym.shape,len(test_sym)))
                    #print("test_bits_estimated:{} test_bits_real:{}".format(test_bits_estimated,test_bits_real))
                    test_mse_loss = mse(test_reconstructions, test_images)
                    across_batch_acc.append(j, test_bits_estimated, test_bits_real, test_mse_loss)
                    test_bits_estimated_mean, test_bits_real_mean, test_mse_loss_mean = \
                        across_batch_acc.means()
                    test_acc.append(
                    i, test_bits_estimated_mean, test_bits_real_mean, test_mse_loss_mean)
                    plotter.update('Test', test_images, test_reconstructions, test_sym,
                                test_acc, rate_loss_enabled)
                ae.train()
                prob.train()





class RNNConditionalProbabilityModel(nn.Module):
  def __init__(self, bottleneck_dim=32):
    super(RNNConditionalProbabilityModel, self).__init__()
    self.model = PixelRNN(num_layers=3, hidden_dims=64, input_size=2)
    self.coder = AECompressor_RNN()


  def process_prob(self,prob):
    #probability =  {1: Range(0, 2048), 0: Range(2048, 4096)}
    prob_list = []
    for i in range(len(prob)):
      
      prob_0_scaled = 4096 - int(prob[i] * 4096)
      p =  {1: Range(0, prob_0_scaled), 0: Range(prob_0_scaled, 4096)}
      prob_list.append(p)

    return prob_list 

  def forward(self, sym):
   

    # Get the output of the CNN.
    sym = torch.clip(sym,min=0,max=1)
    output_prob = self.model(sym.float())
    output_prob = output_prob.flatten() #[batch*32*2*2]
    sym = sym.flatten().tolist() #[batch*32*2*2]
    probability = self.process_prob(output_prob)
    #sym = sym.to(dtype=torch.uint8)
    #print("sym:{} probability:{}".format(len(sym),len(probability)))
    compressed_sym = self.coder.compress(sym,probability)
    decompressed_sym = self.coder.decompress(compressed_sym,probability)
    #print("original len:{} compressed_sym:{}".format(len(sym),len(compressed_sym)))
    #print("sym:{}  decompressed:{}".format(sym[:10],decompressed_sym[:10]))
    #print("Same:",(sym==decompressed_sym))
    estimated_bits, real_bits = torch.tensor(len(sym)), torch.tensor(len(compressed_sym))
    #print("sym:{} decoded_byte_stream:{}".format(sym.shape,decoded_byte_steam.shape))
    return estimated_bits, real_bits






def main():
  p = argparse.ArgumentParser()
  p.add_argument('--max_training_itr', type=int)
  flags = p.parse_args()
  train_test_loop(max_training_itr=flags.max_training_itr)


if __name__ == '__main__':
  main()