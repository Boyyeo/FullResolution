### For one-shot lstm
CUDA_VISIBLE_DEVICES=0 python train.py --bs 32 --lr 0.0005 --out_dir 'output/' --name 'lstm-oneshot' --data_path 'CIFAR-10-images/train' --save_record True --iterations 16 --recon_fw 'one-shot' --arch 'lstm'

### For additive lstm
CUDA_VISIBLE_DEVICES=0 python train.py --bs 32 --lr 0.0005 --out_dir 'output/' --name 'lstm-additive' --data_path 'CIFAR-10-images/train' --save_record True --iterations 16 --recon_fw 'one-shot' --arch 'lstm'
