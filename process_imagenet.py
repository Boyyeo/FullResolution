import numpy as np 
import torch 
from torchvision.utils import save_image
def load_databatch(filename, img_size=32):

    d = np.load(filename)
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/255.
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    #x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=X_train.astype(np.float32),
        Y_train=Y_train.astype(np.int32),
        mean=mean_image)

data_dict = load_databatch('Imagenet32_train_npz/train_data_batch_1.npz')
print("X:{} Y:{}".format(data_dict['X_train'].shape,data_dict['Y_train'].shape))

for i,img in enumerate(data_dict['X_train']):
    img = torch.from_numpy(img)
    save_image(img,'img_{}.jpg'.format(str(i).zfill(6)))
    if i > 60 :
        break
