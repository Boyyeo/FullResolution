### For one-shot lstm
CUDA_VISIBLE_DEVICES=0 python train.py --bs 32 --lr 0.0005 --out_dir 'output/' --name 'lstm-oneshot' --data_path 'CIFAR-10-images/train' --save_record True --iterations 16 --recon_fw 'one-shot' --arch 'lstm'

### For additive lstm
CUDA_VISIBLE_DEVICES=0 python train.py --bs 32 --lr 0.0005 --out_dir 'output/' --name 'lstm-additive' --data_path 'CIFAR-10-images/train' --save_record True --iterations 16 --recon_fw 'additive' --arch 'lstm'

### For residual_scaling lstm
CUDA_VISIBLE_DEVICES=0 python train.py --bs 32 --lr 0.0005 --out_dir 'output/' --name 'lstm-resscale' --data_path 'CIFAR-10-images/train' --save_record True --iterations 16 --recon_fw 'residual-scaling' --arch 'lstm'

### For Evaluation 
CUDA_VISIBLE_DEVICES=0 python train.py  --recon_fw 'one-shot' --arch 'lstm' --epoch 0 --save_record True --resume_ckpt output/lstm-additive-20231224-045144/checkpoint/model_0.pyt
CUDA_VISIBLE_DEVICES=0 python train.py  --recon_fw 'residual-scaling' --arch 'lstm' --epoch 0 --save_record True 


