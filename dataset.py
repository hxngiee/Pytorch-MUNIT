import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
import os


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, data_type='float32', transform=[], mode='train'):

        self.data_dir_a = data_dir + 'A'
        self.data_dir_b = data_dir + 'B'
        self.transform = transform
        self.data_type = data_type
        self.mode = mode

        data_name = data_dir.split('/')[-1]     # linux
        # data_name = data_dir.split('\\')[-1]  # window?

        if os.path.exists(self.data_dir_a):
            lst_data_a = os.listdir(self.data_dir_a)
            lst_data_a = [f for f in lst_data_a if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]
            lst_data_a.sort()
        else:
            lst_data_a = []

        if os.path.exists(self.data_dir_b):
            lst_data_b = os.listdir(self.data_dir_b)
            lst_data_b = [f for f in lst_data_b if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]
            lst_data_b.sort()
        else:
            lst_data_b = []

        self.lst_data_a = lst_data_a
        self.lst_data_b = lst_data_b


    def __getitem__(self, index):
        data = {}

        data_a = plt.imread(os.path.join(self.data_dir_a, self.lst_data_a[index]))[:,:,:3]
        if data_a.ndim == 2:
            data_a = data_a[:,:,np.newaxis]
        if data_a.dtype == np.uint8:
            data_a = data_a / 255.0
        data['data_a'] = data_a

        data_b = plt.imread(os.path.join(self.data_dir_b, self.lst_data_b[index]))[:,:,:3]
        if data_b.ndim == 2:
            data_b = data_b[:,:,np.newaxis]
        if data_b.dtype == np.uint8:
            data_b = data_b / 255.0
        data['data_b'] = data_b

        if self.transform:
            data['data_a'] = self.transform(data['data_a'])
            data['data_b'] = self.transform(data['data_b'])
            # data = self.transform(data)

        return data


    def __len__(self):
        if len(self.lst_data_a) < len(self.lst_data_b):
            return len(self.lst_data_a)
        else:
            return len(self.lst_data_b)


class ToTensor(object):
    def __call__(self, data):
        data = data.transpose((2, 0, 1)).astype(np.float32)
        return torch.from_numpy(data)


class Normalize(object):
    def __call__(self, data):
        data = 2 * data - 1

        # print(data)
        # data['data_a'] = 2*data['data_a'] - 1
        # domainB에 대해서도 전체리해주는게 맞나?
        # data['data_b'] = 2*data['data_b'] - 1

        return data


class RandomFlip(object):
    def __call__(self, data):
        if np.random.rand() > 0.5:
            data = np.fliplr(data)

        return data


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        if isinstance(self.output_size, int):
          if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
          else:
            new_h, new_w = self.output_size, self.output_size * w / h
        else:
          new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        data = transform.resize(data, (new_h, new_w))
        return data


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        #img2img
        # keys = list(data.keys())
        # h, w = data[keys[0]].shape[:2]
        h, w = data.shape[:2]
        new_h, new_w = self.output_size

        top = int(abs(h - new_h) / 2)
        left = int(abs(w - new_w) / 2)

        data = data[top: top + new_h, left: left + new_w]

        # for key, value in data.items():
        #     data[key] = value[top:top+new_h,left:left+new_w]
        # print(data['data_a'].shape)

        return data


class RandomCrop(object):

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        data = data[top: top + new_h, left: left + new_w]
        return data


class ToNumpy(object):
    def __call__(self, data):

        if data.ndim == 3:
            data = data.to('cpu').detach().numpy().transpose((1, 2, 0))
        elif data.ndim == 4:
            data = data.to('cpu').detach().numpy().transpose((0, 2, 3, 1))

        return data


class Denomalize(object):
    def __call__(self, data):

        return (data + 1) / 2
