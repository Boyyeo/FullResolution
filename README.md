# Full Resolution Image Compression with Recurrent Neural Networks

#### To train Compression Model (Encoder, Binarizer, Decoder) 
```
CUDA_VISIBLE_DEVICES=0 python train.py --bs 128 --lr 0.0005 --out_dir 'output/' --name 'fullres-compress' --data_path 'CIFAR10-images/train' --save_record False --iterations 16
```

#### To train Entropy Coding (Still in Progress)
```
CUDA_VISIBLE_DEVICES=0 python pixel_code.py
```

#### For test arithmetic coding (Still in Progress)
```
python main.py
```

#### To download Datasets
##### CIFAR-10 Directly 
```
git clone https://github.com/YoongiKim/CIFAR-10-images.git
```
##### ImageNet-32x32  (https://image-net.org/download-images.php)
```

```
#### Setup Conda Environment 
```
conda create --name fullres python=3.10
pip install -r requirements.txt
```

