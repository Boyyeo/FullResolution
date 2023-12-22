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
flags.DEFINE_string('recon_framework', "one-shot", "[one-shot,additive,residual-scaling]")
flags.DEFINE_string('architecture', "lstm", "[lstm,gru,residual-gru,associative-lstm]")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    train_transform = T.Compose(
            [
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.ImageFolder(FLAGS.data_path,transform=train_transform) #没有transform，先看看取得的原始图像数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.bs,shuffle=True, num_workers=4)
    net_Enc = EncoderCell().to(device)
    net_Binarizer = Binarizer().to(device)
    net_Dec = DecoderCell().to(device)
    optimizer = optim.Adam(list(net_Enc.parameters()) + list(net_Binarizer.parameters()) + list(net_Dec.parameters()),lr=FLAGS.lr)
    scheduler = MultiStepLR(optimizer, milestones=[3, 10, 20, 50, 100], gamma=0.5)

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


            for _ in range(FLAGS.iterations):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = net_Enc(res, encoder_h_1, encoder_h_2, encoder_h_3)
                codes = net_Binarizer(encoded)
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = net_Dec(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                res = res - output
                losses.append(res.abs().mean())

            loss = sum(losses) / FLAGS.iterations
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        epoch_loss /= len(train_loader)
        print("Epoch {}/{} Loss:{} ".format(epoch,FLAGS.epoch,round(epoch_loss,6)))



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
        run = 0
        while os.path.exists(FLAGS.out_dir + FLAGS.name + str(run) + "/"):
            run += 1
        FLAGS.out_dir = FLAGS.out_dir + FLAGS.name + str(run) + "/"
        os.mkdir(FLAGS.out_dir)
        os.mkdir(FLAGS.out_dir + "checkpoint")
    train()


if __name__ == '__main__':
    app.run(main)
