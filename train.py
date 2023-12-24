from absl import flags, app
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms as T
from absl import flags, app
import random
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
import torchvision.datasets as datasets
from network import * 
from torch.autograd import Variable
from tqdm import tqdm 
from datetime import datetime
import requests

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, "random seed")
flags.DEFINE_integer('bs', 128, "batch size")
flags.DEFINE_integer('epoch', 500, "number of epochs")
flags.DEFINE_float('lr', 0.0005, 'learning rate for training children')
flags.DEFINE_float('weight_decay', 1e-5, 'weight decay for optimizer')
flags.DEFINE_string('out_dir', "output/", "Folder output sample_image")
flags.DEFINE_string('name', "experiment", "Folder output sample_image")
flags.DEFINE_string('data_path', "CIFAR-10-images/train", "Folder of dataset images")
flags.DEFINE_boolean('save_record', False, "Whether to save the experiment results")
flags.DEFINE_integer('iterations', 16, "number of iteration of RNN in training")
flags.DEFINE_string('recon_framework', "one-shot", "[one-shot, additive, residual-scaling]")
flags.DEFINE_string('architecture', "lstm", "[lstm, gru, residual-gru, associative-lstms]")


class Networks:
    def __init__(self, arch="lstm", recon_fw="one-shot"):
        self.Enc = EncoderCell()
        self.Binarizer = Binarizer()
        self.Dec = DecoderCell()
        self.recon_fw = recon_fw
        self.arch = arch
        if recon_fw == "residual-scaling":
            self.GainEstimator = GainEstimatorCell()


    def save(self, idx, dir):
        torch.save(self.Enc.state_dict(), f'{dir}/encoder_epoch_{idx}.pth')
        torch.save(self.Binarizer.state_dict(), f'{dir}/binarizer_epoch_{idx}.pth')
        torch.save(self.Dec.state_dict(), f'{dir}/decoder_epoch_{idx}.pth')


def zero_order_hold_upsampling(x):
    # [32, 1, 2, 2] -> [32, 1, 32, 32]
    upsampled = F.interpolate(x, scale_factor=16)
    # print(f'upsampled shape: {upsampled.shape}')
    return upsampled


def train(save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    train_transform = T.Compose(
            [
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.ImageFolder(FLAGS.data_path,transform=train_transform) #没有transform，先看看取得的原始图像数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.bs,shuffle=True, num_workers=4)

    net = Networks(FLAGS.architecture, FLAGS.recon_framework)
    net.Enc = net.Enc.to(device)
    net.Binarizer = net.Binarizer.to(device)
    net.Dec = net.Dec.to(device)
    if net.recon_fw == "residual-scaling":
        net.GainEstimator = net.GainEstimator.to(device)
    optimizer = optim.Adam(list(net.Enc.parameters()) + list(net.Binarizer.parameters()) + list(net.Dec.parameters()),lr=FLAGS.lr)
    # scheduler = MultiStepLR(optimizer, milestones=[3, 10, 20, 50, 100], gamma=0.5)
    with torch.autograd.detect_anomaly(): 
        for epoch in range(FLAGS.epoch):
            epoch_loss = 0.0
            for x,_ in tqdm(train_loader):
                x = x.to(device)
                ## init lstm state
                encoder_h_1 = (Variable(torch.zeros(x.size(0), 256, 8, 8).cuda()),
                            Variable(torch.zeros(x.size(0), 256, 8, 8).cuda()))
                encoder_h_2 = (Variable(torch.zeros(x.size(0), 512, 4, 4).cuda()),
                            Variable(torch.zeros(x.size(0), 512, 4, 4).cuda()))
                encoder_h_3 = (Variable(torch.zeros(x.size(0), 512, 2, 2).cuda()),
                            Variable(torch.zeros(x.size(0), 512, 2, 2).cuda()))

                decoder_h_1 = (Variable(torch.zeros(x.size(0), 512, 2, 2).cuda()),
                            Variable(torch.zeros(x.size(0), 512, 2, 2).cuda()))
                decoder_h_2 = (Variable(torch.zeros(x.size(0), 512, 4, 4).cuda()),
                            Variable(torch.zeros(x.size(0), 512, 4, 4).cuda()))
                decoder_h_3 = (Variable(torch.zeros(x.size(0), 256, 8, 8).cuda()),
                            Variable(torch.zeros(x.size(0), 256, 8, 8).cuda()))
                decoder_h_4 = (Variable(torch.zeros(x.size(0), 128, 16, 16).cuda()),
                            Variable(torch.zeros(x.size(0), 128, 16, 16).cuda()))
                
                patches = Variable(x.cuda())
                optimizer.zero_grad()

                losses = []
                res = patches - 0.5
                res_0 = res
                recon_imgs = 0 # recon_imgs shape: [32, 3, 32, 32] [batch size, channel, w, h]
                gain = torch.ones((32, 1, 2, 2)).to(device)
                # print(f'gain initialize: {gain}')
                # print(f'gain shape initialize: {gain.shape}')

                for _ in range(FLAGS.iterations):
                    print(f'iteration {_}')
                    if net.recon_fw == "residual-scaling":
                        upsampled_gain = zero_order_hold_upsampling(gain)
                        enc_input = torch.mul(res, upsampled_gain) # res multiply ZOH(gain)
                        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = net.Enc(enc_input, encoder_h_1, encoder_h_2, encoder_h_3)
                        codes = net.Binarizer(encoded)
                        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = net.Dec(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                        res_bar = torch.div(output, upsampled_gain) # output divide ZOH(gain)

                        recon_imgs += res_bar
                        # update gain factor
                        gain = net.GainEstimator(recon_imgs)

                    else:
                        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = net.Enc(res, encoder_h_1, encoder_h_2, encoder_h_3)
                        codes = net.Binarizer(encoded)
                        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = net.Dec(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                        
                        if net.recon_fw == "one-shot":
                            recon_imgs = output
                        elif net.recon_fw == "additive":
                            recon_imgs += output
                                    
                    res = res_0 - recon_imgs
                    losses.append(res.abs().mean())

                loss = sum(losses) / FLAGS.iterations
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            epoch_loss /= len(train_loader)
            print("Epoch {}/{} Loss:{} ".format(epoch,FLAGS.epoch,round(epoch_loss,6)))
            if epoch%10 == 0:
                send_message("Epoch {}/{} Loss:{} ".format(epoch,FLAGS.epoch,round(epoch_loss,6)))

            if FLAGS.save_record:
                net.save(epoch, save_dir)




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.save_record:
        if not os.path.exists(FLAGS.out_dir):
            os.mkdir(FLAGS.out_dir)
        now = datetime.now() # current date and time
        date_time = now.strftime("%Y%m%d-%H%M%S")   
        FLAGS.out_dir = FLAGS.out_dir + FLAGS.name + date_time + "/"
        os.mkdir(FLAGS.out_dir)
        os.mkdir(FLAGS.out_dir + "checkpoint")
        save_dir = FLAGS.out_dir + "checkpoint"
        train(save_dir)
    else:
        train()


def send_message(message = ''):
    headers = {"Authorization": "Bearer " + 'TyIF7OLhJ1oy1UKowBnPRSWi5aDlyGYTbDg1tolnoAe'}
    data = { 'message': message }
    requests.post("https://notify-api.line.me/api/notify", headers = headers, data = data)



if __name__ == '__main__':
    app.run(main)