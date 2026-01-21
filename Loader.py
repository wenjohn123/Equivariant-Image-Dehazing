from distutils.command.clean import clean

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import  os
import h5py
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class HazyDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path

        assert os.path.isdir(dir_path), "{} is not a directory".format(dir_path)
        hazy_path = os.path.join(dir_path, 'hazy')
        self.hazy_files = []
        for filename in sorted(os.listdir(hazy_path)):
            if filename.endswith('.png'):
                self.hazy_files.append(os.path.join(hazy_path, filename))
        self.transform = transform
        print("len of clean: ", len(self.hazy_files))
    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy_img = self.hazy_files[idx]
        hazy_img = Image.open(hazy_img).convert('RGB')

        if self.transform:
            transform1 = transforms.Compose([
                transforms.Resize((256, 384), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor()
            ])
            hazy_img = self.transform(hazy_img)
        else:
            hazy_img = torch.FloatTensor(hazy_img)
        result=hazy_img
        return result

def get_dataloader(dir_path, batch_size=1, shuffle=True,num_workers=0, transform=None):
    dataset = HazyDataset(dir_path, transform)
    print("shuffle: ", shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader