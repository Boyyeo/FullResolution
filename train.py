from absl import flags, app
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from absl import flags, app
import random
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision.utils import save_image
import os
import torchvision.datasets as datasets
from network import * 
from torch.autograd import Variable
from tqdm import tqdm 
from datetime import datetime
from metric import ms_ssim_torch
from piq import ssim, psnr
from PIL import Image 
import requests
import time
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, "random seed")
flags.DEFINE_integer('bs', 128, "batch size")
flags.DEFINE_integer('epoch', 500, "number of epochs")
flags.DEFINE_float('lr', 0.0005, 'learning rate for training children')
flags.DEFINE_float('weight_decay', 1e-5, 'weight decay for optimizer')
flags.DEFINE_string('out_dir', "output/", "Folder output sample_image")
flags.DEFINE_string('name', "experiment", "Folder output sample_image")
flags.DEFINE_string('data_path', "CIFAR-10-images/train", "Folder of dataset images")
flags.DEFINE_string('eval_path', "test/images", "Folder of dataset images for evaluation")
flags.DEFINE_boolean('save_record', False, "Whether to save the experiment results")
flags.DEFINE_integer('iterations', 16, "number of iteration of RNN in training")
flags.DEFINE_string('recon_fw', "one-shot", "[one-shot, additive, residual-scaling]")
flags.DEFINE_string('arch', "lstm", "[lstm, gru, residual-gru, associative-lstms]")
flags.DEFINE_string('resume_ckpt', None, "path to checkpoint")



def zero_order_hold_upsampling(x,h,w):
    # [32, 1, 2, 2] -> [32, 1, 32, 32]
    upsampled = F.interpolate(x, size=(h,w))
    # print(f'upsampled shape: {upsampled.shape}')
    return upsampled


class Networks:
    def __init__(self, arch="lstm", recon_fw="one-shot"):
        self.Enc = EncoderCell(arch)
        self.Binarizer = Binarizer()
        self.Dec = DecoderCell(arch)
        self.recon_fw = recon_fw
        if recon_fw == "residual-scaling":
            self.GainEstimator = GainEstimatorCell()
        else:
            self.GainEstimator = None 
        self.arch = arch

    def save(self, idx, dir):
        checkpoint = {
                'enc': self.Enc.state_dict(),
                'binarizer': self.Binarizer.state_dict(),
                'dec':self.Dec.state_dict(),
                'epoch':idx,
                'gain': self.GainEstimator}
        
        torch.save(checkpoint,'{}/model_{}.pyt'.format(dir,'last'))

    def load(self):
        checkpoint = torch.load(FLAGS.resume_ckpt)
        self.Enc.load_state_dict(checkpoint['enc'])
        self.Binarizer.load_state_dict(checkpoint['binarizer'])
        self.Dec.load_state_dict(checkpoint['dec'])
        if self.GainEstimator is not None: 
            self.GainEstimator.load_state_dict(checkpoint['gain'])

        return checkpoint['epoch']


def train(save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    net = Networks(FLAGS.arch, FLAGS.recon_fw)
    if FLAGS.resume_ckpt is not None:
        start_epoch = net.load()
        print("The network is successfully loaded from checkpoint with epoch {}".format(start_epoch))
    else:
        start_epoch = 0

    net.Enc = net.Enc.to(device)
    net.Binarizer = net.Binarizer.to(device)
    net.Dec = net.Dec.to(device)
    if net.recon_fw == "residual-scaling":
        net.GainEstimator = net.GainEstimator.to(device)
        optimizer = optim.Adam(list(net.GainEstimator.parameters()) + list(net.Enc.parameters()) + list(net.Binarizer.parameters()) + list(net.Dec.parameters()),lr=FLAGS.lr)
    else:
        optimizer = optim.Adam(list(net.Enc.parameters()) + list(net.Binarizer.parameters()) + list(net.Dec.parameters()),lr=FLAGS.lr)
    #scheduler = MultiStepLR(optimizer, milestones=[3, 10, 20, 50, 100], gamma=0.5)
    
    if FLAGS.epoch <= 0:
        eval(net, -1)

    train_transform = T.Compose(
            [T.RandomHorizontalFlip(),
             T.ToTensor()])

    train_dataset = datasets.ImageFolder(FLAGS.data_path,transform=train_transform) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.bs,shuffle=True, num_workers=4)

    for epoch in range(start_epoch,FLAGS.epoch):
        epoch_loss = 0.0
        
        for x,_ in tqdm(train_loader):
            x = x.to(device)
            ## init lstm state
            encoder_h_1 = (Variable(torch.zeros(x.size(0), 256, 8, 8).cuda()),Variable(torch.zeros(x.size(0), 256, 8, 8).cuda()))
            encoder_h_2 = (Variable(torch.zeros(x.size(0), 512, 4, 4).cuda()),Variable(torch.zeros(x.size(0), 512, 4, 4).cuda()))
            encoder_h_3 = (Variable(torch.zeros(x.size(0), 512, 2, 2).cuda()),Variable(torch.zeros(x.size(0), 512, 2, 2).cuda()))

            decoder_h_1 = (Variable(torch.zeros(x.size(0), 512, 2, 2).cuda()),Variable(torch.zeros(x.size(0), 512, 2, 2).cuda()))
            decoder_h_2 = (Variable(torch.zeros(x.size(0), 512, 4, 4).cuda()),Variable(torch.zeros(x.size(0), 512, 4, 4).cuda()))
            decoder_h_3 = (Variable(torch.zeros(x.size(0), 256, 8, 8).cuda()),Variable(torch.zeros(x.size(0), 256, 8, 8).cuda()))
            decoder_h_4 = (Variable(torch.zeros(x.size(0), 128, 16, 16).cuda()),Variable(torch.zeros(x.size(0), 128, 16, 16).cuda()))
    
            patches = Variable(x.cuda())
            optimizer.zero_grad()

            losses = []
            res = patches - 0.5
            res_0 = res
            recon_imgs = torch.zeros_like(res_0).to(device) # recon_imgs shape: [32, 3, 32, 32] [batch size, channel, w, h]

            for it in range(FLAGS.iterations):
                if net.recon_fw == "residual-scaling":
                    # update gain factor
                    if it != 0 :
                        gain = net.GainEstimator(res_bar)
                        upsampled_gain = zero_order_hold_upsampling(gain,h=res_bar.shape[-2],w=res_bar.shape[-1])
                        #print("upsampled_gain:{} res:{} res_bar:{} gain:{}".format(upsampled_gain.shape,res.shape,res_bar.shape,gain.shape))
                    else:
                        upsampled_gain = torch.ones((res.shape[0], 1, 32, 32)).to(device)

                    enc_input = torch.mul(res, upsampled_gain) # res multiply ZOH(gain)
                   
                else:
                    enc_input = res

                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = net.Enc(enc_input, encoder_h_1, encoder_h_2, encoder_h_3)
                codes = net.Binarizer(encoded)

                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = net.Dec(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                
                if net.recon_fw == "residual-scaling":
                    res_bar = torch.div(output, upsampled_gain) # output divide ZOH(gain)
                    recon_imgs += res_bar
                elif net.recon_fw == 'one-shot':
                    recon_imgs = output
                elif net.recon_fw == 'additive':
                    recon_imgs += output
                                
                res = res_0 - recon_imgs
                losses.append(res.abs().mean())

            loss = sum(losses) / FLAGS.iterations
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_loader)
        print("Epoch {}/{} Loss:{} ".format(epoch,FLAGS.epoch,round(epoch_loss,6)))

        ###### EVALUATION ######
        if FLAGS.save_record:
            net.save(epoch, save_dir)
            if epoch % 10 == 0:
                eval(net,epoch)


@torch.no_grad()
def eval(net, epoch):
    save_folder_path = '{}/eval_kodak/epoch_{}/'.format(FLAGS.out_dir,str(epoch).zfill(4))
    os.makedirs(save_folder_path,exist_ok=True)
    os.makedirs('results/',exist_ok=True)

    ##### EVALUATION ####
    test_image_paths =  os.listdir(FLAGS.eval_path)
    test_image_paths.sort()
    test_image_paths = [os.path.join(FLAGS.eval_path,path) for path in test_image_paths]
    i = 1
    for test_path in tqdm(test_image_paths):
        save_img_folder = save_folder_path + '{}/'.format(test_path.split('/')[-1].split('.png')[0]) #e.g output/exp/eval_kodak/epoch_0/kodim12/
        os.makedirs(save_img_folder,exist_ok=True)
        print("Evluating image:{} ...".format(test_path))

        image = T.ToTensor()(Image.open(test_path).convert('RGB')).unsqueeze(0).cuda()
        batch_size, C, height, width = image.shape
        encoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda(), Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda())
        encoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda(),Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda())
        encoder_h_3 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda(),Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda())

        decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda(),Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda())
        decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda(),Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda())
        decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda(),Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda())
        decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2)).cuda(),Variable(torch.zeros(batch_size, 128, height // 2, width // 2)).cuda())
    
        codes = []
        res = image - 0.5
        res_0 = res
        recon_imgs = torch.zeros(1, 3, height, width).cuda()
        losses = []
        #psnr_list, ssim_list, ms_ssim_list, psnr_hvs_list = [],[],[],[] 
        results = []
        print("--------------------------------------------------------------------------------")
        for iters in range(FLAGS.iterations):
            if net.recon_fw == "residual-scaling":
                # update gain factor
                if iters != 0 :
                    gain = net.GainEstimator(res_bar)
                    upsampled_gain = zero_order_hold_upsampling(gain,h=res_bar.shape[-2],w=res_bar.shape[-1])
                else:
                    upsampled_gain = torch.ones((batch_size, 1, height, width)).cuda()

                enc_input = torch.mul(res, upsampled_gain) # res multiply ZOH(gain)
            else:
                enc_input = res

            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = net.Enc(enc_input, encoder_h_1, encoder_h_2, encoder_h_3)
            code = net.Binarizer(encoded)
            start_decode = time.time()
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = net.Dec(code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            end_decode = time.time()
            codes.append(code.data.cpu().numpy())

            if net.recon_fw == "residual-scaling":
                res_bar = torch.div(output, upsampled_gain) # output divide ZOH(gain)
                recon_imgs += res_bar
            elif net.recon_fw == 'one-shot':
                recon_imgs = output
            elif net.recon_fw == 'additive':
                recon_imgs += output
                            
            res = res_0 - recon_imgs
            losses.append(res.abs().mean())

            saved_recon_imgs = torch.clip(recon_imgs + 0.5,min=0.0,max=1.0).data.cpu()
            save_image(saved_recon_imgs,'{}/iter_{}.png'.format(save_img_folder,str(iters).zfill(2)))
            psnr_index, ssim_index, ms_ssim_index = psnr(image.cpu(),saved_recon_imgs), ssim(image.cpu(),saved_recon_imgs), ms_ssim_torch(image.cpu(),saved_recon_imgs)
            print("Iter:{} PSNR:{} SSIM:{} MS_SSIM:{} Decoding Time:{}".format(str(iters).zfill(2),round(psnr_index.item(),3),round(ssim_index.item(),3),round(ms_ssim_index.item(),3),round(end_decode-start_decode,6)))
            results.append([round(psnr_index.item(),3),round(ssim_index.item(),3),round(ms_ssim_index.item(),3),round(end_decode-start_decode,6)])

        loss = sum(losses) / FLAGS.iterations
        codes = (np.stack(codes).astype(np.int8) + 1) // 2
        print("LOSS:{} IMAGE SHAPE:{} COMPRESSED FULL CODE LENGTH:{}".format(round(loss.item(),4),image.shape,code.shape))
        print("--------------------------------------------------------------------------------")

        results_df = pd.DataFrame(results)
        file_name = 'results/result_' + str(i) + '.csv'
        results_df.to_csv(file_name, header=['PSNR', 'SSIM', 'MS_SSIM', 'Decoding Time'])
        i += 1
        #imsave( os.path.join(args.output, '{:02d}.png'.format(iters)), np.squeeze(image.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0))
        #print("codes:",codes.shape)
        #export = np.packbits(codes.reshape(-1))
        #np.savez_compressed(args.output, shape=codes.shape, codes=export)
        

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
        FLAGS.out_dir = FLAGS.out_dir + FLAGS.name + '-' + date_time + "/"
        os.mkdir(FLAGS.out_dir)
        os.mkdir(FLAGS.out_dir + "checkpoint")
        save_dir = FLAGS.out_dir + "checkpoint"
        print("all output will be saved to folder: ",FLAGS.out_dir)
        train(save_dir)
    else:
        train()


# def send_message(message = ''):
#     headers = {"Authorization": "Bearer " + 'TyIF7OLhJ1oy1UKowBnPRSWi5aDlyGYTbDg1tolnoAe'}
#     data = { 'message': message }
#     requests.post("https://notify-api.line.me/api/notify", headers = headers, data = data)



if __name__ == '__main__':
    app.run(main)
