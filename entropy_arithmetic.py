import argparse
import dataclasses
import itertools
import warnings

import os
import time

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import matplotlib
from pixelrnn import PixelRNN 
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
                    lr_ae=1e-4,
                    lr_rnn=1e-4,
                    num_test_batches=10,
                    max_training_itr=None,
                    mnist_download_dir='data',
                    ae_total_epoch = 10,
                    rnn_total_epoch = 50,
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

  mse = nn.MSELoss() ## Loss function of autoentropy
  optim_net_AE = torch.optim.Adam(ae.parameters(),lr=lr_ae)



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

  test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                mnist_download_dir, train=False, download=True, transform=transforms),
                batch_size=batch_size, shuffle=True)

  for epoch in range(ae_total_epoch):
    train_mse_avg = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optim_net_AE.zero_grad()
        reconstructions, sym = ae(images)
        mse_loss = mse(reconstructions, images).mean()
        loss = mse_loss
        loss.backward()
        train_mse_avg += loss.item()
        optim_net_AE.step()

    ae.eval()
    test_mse_avg = 0.0
    with torch.no_grad():
        for j, (test_images, test_labels) in enumerate(test_loader):
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_reconstructions, test_sym = ae(test_images)
            #print("test_sym:{} length:{}".format(test_sym.shape,len(test_sym)))
            #print("test_bits_estimated:{} test_bits_real:{}".format(test_bits_estimated,test_bits_real))
            test_mse_loss = mse(test_reconstructions, test_images)
            test_mse_avg += test_mse_loss.item() 
        train_mse_avg /=  len(train_loader)
        test_mse_avg  /=  len(test_loader)
        print("EPOCH {}/{} train mse_loss:{} test mse_loss:{}".format(epoch,ae_total_epoch,train_mse_avg,test_mse_avg))
    ae.train()


  #############################################################################################################################
  print("Finished Training AutoEncoder ---> Training RNN Probabilistic Model")
  ae.eval()
  for param in ae.parameters():
      param.requires_grad = False

  prob = prob.to(device)
  optim_net_RNN = torch.optim.Adam(prob.parameters(),lr=lr_rnn)
  bce = nn.BCELoss()
  for epoch in range(rnn_total_epoch):
      train_bce_avg = 0.0
      train_compressed_bit_avg = 0
      for images, labels in tqdm(train_loader):
          images = images.to(device)
          labels = labels.to(device)

          optim_net_RNN.zero_grad()
          reconstructions, sym = ae(images)
          pred_prob, target_prob, bits_data_length, bits_compressed_length = prob(sym.detach())
          train_compressed_bit_avg += bits_compressed_length.item()
          loss = bce(pred_prob,target_prob)
          loss.backward()
          train_bce_avg += loss.item()
          #print("shape:{} {}".format((pred_prob.round() == target_prob).shape,pred_prob.shape[0]))
          print("unique pred:{} [0:{},1:{}] sym:{} [0:{},1:{}]".format(torch.unique(pred_prob.round()),(pred_prob.round()==0).sum(),(pred_prob.round()==1).sum(),torch.unique(target_prob),(target_prob==0).sum(),(target_prob==1).sum()))
        
          accuracy = (pred_prob.round() == target_prob).sum() / pred_prob.shape[0]
          optim_net_RNN.step()
          print("accuracy:{} loss:{} data-bit:{} compress-bit:{}".format(accuracy.item(),loss.item(),bits_data_length.item(), bits_compressed_length.item()))

      prob.eval()
      test_bce_avg = 0.0
      test_compressed_bit_avg = 0
      with torch.no_grad():
          for j, (test_images, test_labels) in enumerate(test_loader):
              test_images = test_images.to(device)
              test_labels = test_labels.to(device)
              test_reconstructions, test_sym = ae(test_images)
              pred_prob, target_prob, bits_data_length, test_bits_compressed_length = prob(test_sym)
              test_compressed_bit_avg += test_bits_compressed_length.item()

              test_bce_loss = bce(pred_prob,target_prob)
              test_bce_avg += test_bce_loss.item()  
          train_bce_avg /=  len(train_loader)
          test_bce_avg  /=  len(test_loader)
          train_compressed_bit_avg /=  len(train_loader)
          test_compressed_bit_avg  /=  len(test_loader)
          print("EPOCH {}/{} train bce_loss:{} test bce_loss:{} data-bit:{} train-bit:{} test-bit:{}".format(epoch,ae_total_epoch,train_bce_avg,test_bce_avg, bits_data_length.item(),train_compressed_bit_avg,test_compressed_bit_avg))
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
      
      prob_0_scaled = int(prob[i] * 4096)
      p =  {1: Range(0, prob_0_scaled), 0: Range(prob_0_scaled, 4096)}
      prob_list.append(p)

    return prob_list 

  def forward(self, sym):
   
    # Get the output of the CNN.
    sym = torch.clip(sym,min=0,max=1)
    input_sym = sym.flatten().float()
    output_prob = self.model(sym.float())
    output_prob = output_prob.flatten() #[batch*32*2*2]
    sym = sym.flatten().tolist() #[batch*32*2*2]
    probability = self.process_prob(output_prob)
    compressed_sym = self.coder.compress(sym,probability)
    decompressed_sym = self.coder.decompress(compressed_sym,probability)
    estimated_bits, real_bits = torch.tensor(len(sym)), torch.tensor(len(compressed_sym))
    assert (sym == decompressed_sym)
    return output_prob.unsqueeze(1),input_sym.unsqueeze(1), estimated_bits, real_bits


class STEQuantize(torch.autograd.Function):
  """Straight-Through Estimator for Quantization.

  Forward pass implements quantization by rounding to integers,
  backward pass is set to gradients of the identity function.
  """
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return x.round()

  @staticmethod
  def backward(ctx, grad_outputs):
    return grad_outputs


class Autoencoder(nn.Module):
  def __init__(self, bottleneck_size, L):
    if L % 2 != 1:
      raise ValueError(f'number of levels L={L}, must be odd number!')
    super(Autoencoder, self).__init__()
    self.L = L
    self.enc = nn.Sequential(
      nn.Conv2d(1, 32, 5, stride=2, padding=2),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, 5, stride=2, padding=2),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, 5, stride=2, padding=2),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, 5, stride=2, padding=2),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, bottleneck_size, 1, stride=1, padding=0, bias=False),
    )

    self.dec = nn.Sequential(
      nn.ConvTranspose2d(bottleneck_size, 32, 5, stride=2, padding=2, output_padding=1),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
      nn.InstanceNorm2d(32),
      nn.ReLU(),

      # Add a few convolutions at the final resolution.
      nn.Conv2d(32, 32, 3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 32, 1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(32, 1, 1, stride=1),
    )

    self.quantize = STEQuantize.apply
    self.bottleneck_shape = (bottleneck_size, 2, 2)

  def forward(self, image):
    # Encode image x into the latent.
    #print("image:",image.shape)
    latent = self.enc(image)
    #print("ae output:",latent.shape)
    # The jiggle is there so that the lowest and highest symbol are not at
    # the boundary. Probably not needed.
    jiggle = 0.2
    spread = self.L - 1 + jiggle
    # The sigmoid clamps to [0, 1], then we multiply it by spread and substract
    # spread / 2, so that the output is nicely centered around zero and
    # in the interval [-spread/2, spread/2]
    latent = torch.sigmoid(latent) * spread - spread / 2
    latent_quantized = self.quantize(latent)
    reconstructions = self.dec(latent_quantized)
    sym = latent_quantized + self.L // 2
    return reconstructions, sym.to(torch.long)

def main():
  p = argparse.ArgumentParser()
  p.add_argument('--max_training_itr', type=int)
  flags = p.parse_args()
  train_test_loop(max_training_itr=flags.max_training_itr)


if __name__ == '__main__':
  main()