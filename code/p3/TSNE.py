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
from model import Convnet, Hallucinator
import matplotlib.pyplot as plt
from sklearn import manifold

filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, data_dir, csv_path, training=False):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.filenames = self.data_df["filename"].tolist()
        self.labels = self.data_df["label"].tolist()
        if training:
            self.transform = transforms.Compose([
                filenameToPILImage,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
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
    
class ClassSampler():
    def __init__(self, batch_num, total_cls , way , N):
        self.batch_num = batch_num
        self.total_cls = total_cls
        self.way = way
        self.N = N
        self.classes = 0
        
        self.iters = []
        batch_size = way
        while len(self.iters) < batch_num:
            self.classes = np.arange(total_cls)
            np.random.shuffle(self.classes)
            for i in range (total_cls // batch_size):
                self.iters.append(self.classes[i * batch_size: (i + 1) * batch_size] )
                if len(self.iters) == batch_num: break
            
    def __len__(self):
        return self.batch_num
    
    def __iter__(self):
        for self.classes in self.iters:
            batch =[]
            for one_class in self.classes:
                img_idx = np.random.randint(0, 600, self.N)
                img_idx = one_class * 600 + img_idx
                batch.append(torch.tensor(img_idx, dtype=torch.int))
                
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            
    
def imshow(image):
    npimg = image.numpy()
    nptran = np.transpose(npimg, (1, 2, 0))
    plt.figure()
    plt.imshow(nptran)
    
    
if __name__=='__main__':
    
    
    csv_path = '..\\..\\hw4_data\\train.csv'
    data_dir = '..\\..\\hw4_data\\train'
    load = './log/M50_20201228_1658/max-acc.pth'
    
    batch_num = 5
    shot = 0
    query = 200
    train_way = 1
    total_cls = 64
    DEVICE = torch.device('cpu')
    
    # TODO: load your model
    DEVICE = torch.device('cpu')
    checkpoint = torch.load(load , map_location = 'cpu')
    
    cnn = Convnet()
    cnn.load_state_dict(checkpoint["cnn"])
    cnn = cnn.to(DEVICE)
    cnn.eval()
    
    hallu = Hallucinator()
    hallu.load_state_dict(checkpoint["hallu"])
    hallu = hallu.to(DEVICE)
    hallu.eval()

    
    trainset = MiniDataset(csv_path=csv_path, data_dir=data_dir)

    train_sampler = ClassSampler(batch_num, total_cls , train_way, shot + query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler)
    
    dataIter_train = train_loader.__iter__()
    
    
    outputs = torch.empty(0, 1600)
    fake_outputs = torch.empty(0, 1600)
    with torch.no_grad():
        for i in range(batch_num):
            (data, _) = dataIter_train.next()
            data = data.to(DEVICE)
            output = cnn(data)
            fake_output = hallu(output)
            
            outputs = torch.cat((outputs, output), dim=0)
            fake_outputs = torch.cat((fake_outputs, fake_output), dim=0)
    
    
    # transform
    tsne = manifold.TSNE(n_components=2, init='random', random_state=5)
    outputs_tsne = tsne.fit_transform(outputs)
    fake_outputs_tsne = tsne.fit_transform(fake_outputs)
    
    
    # draw
    plt.figure(figsize=(10, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, batch_num))
    for i, (X1, X2) in enumerate(zip(outputs_tsne, fake_outputs_tsne)):
        print(i)
        class_idx = int(i/200)
        color = colors[class_idx]
        plt.scatter(X1[0], X1[1], 
                    s=50, marker="X", 
                    color=color, alpha=0.8, linewidths=0.1)
        plt.scatter(X2[0], X2[1], 
                    s=50, marker="^", 
                    color=color, alpha=0.8, linewidths=0.1)
    
    plt.title("tSNE")
    plt.show()
    
    
    
    
    
    