CUDA_VISIBLE_DEVICES=0 python train.py --bs 128 --lr 0.0005 --out_dir 'output/' --name 'fullres-compress' --data_path 'CIFAR10-images/train' --save_record False --iterations 16
CUDA_VISIBLE_DEVICES=1 python mnist_autoencoder.py

