
import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

filenameToPILImage = lambda x: Image.open(x)

# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# random.seed(SEED)
# np.random.seed(SEED)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.filenames = self.data_df["filename"].tolist()
        self.labels = self.data_df["label"].tolist()

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        image = self.transform(os.path.join(self.data_dir, filename))
        return image, label

    def __len__(self):
        return len(self.data_df)


class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)


def imshow(image):
    npimg = image.numpy()
    nptran = np.transpose(npimg, (1, 2, 0))
    plt.figure()
    plt.imshow(nptran)

if __name__ == '__main__':


    trainset = MiniDataset(csv_path='..\\..\\hw4_data\\train.csv',data_dir='..\\..\\hw4_data\\train')

    for i in range(5):
        data = trainset[i]
        # imshow(data[0])
        print(data[1])
    
    print('# images in trainset:', len(trainset))

    dataLoader_train = DataLoader(trainset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=0)
    
    dataIter_train = dataLoader_train.__iter__()
    var = dataIter_train.next()
    imshow(torchvision.utils.make_grid(var[0]))
    print(var[1])
    