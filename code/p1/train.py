import numpy as np
import os
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_1 import MiniDataset
from sampler import CategoriesSampler
from model import Convnet

import time
from torch.utils.tensorboard import SummaryWriter


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n_quary = a.shape[0]
    m_mean = b.shape[0]
    a = a.unsqueeze(1).expand(n_quary, m_mean, -1)
    b = b.unsqueeze(0).expand(n_quary, m_mean, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cosine_similarity(a, b):
    n_quary = a.shape[0]
    m_mean = b.shape[0]
    a = a.unsqueeze(1).expand(n_quary, m_mean, -1)
    b = b.unsqueeze(0).expand(n_quary, m_mean, -1)
    cosine_similarity = F.cosine_similarity(a, b, dim=2, eps=1e-6)
    
    return -(cosine_similarity)


def Manhattan_Distance(a, b):
    n_quary = a.shape[0]
    m_mean = b.shape[0]
    a = a.unsqueeze(1).expand(n_quary, m_mean, -1)
    b = b.unsqueeze(0).expand(n_quary, m_mean, -1)
    logits = -(abs(a - b)).sum(dim=2)
    
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def save_model(name):
    torch.save(model.state_dict(), osp.join(output_path, name + '.pth'))
    
if __name__ == '__main__':
    
    t = time.localtime()
    result = time.strftime("%Y%m%d_%H%M", t)

    # Decide which device we want to run on
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    n_batch = 600
    train_classes = 64
    test_classes = 16
    max_epoch = 200
    save_epoch = 20
    shot= 1
    query= 15
    train_way = 10
    test_way= 5
    output_path = f'log/cosine_similarity{result}'

    
    writer = SummaryWriter(output_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    trainset = MiniDataset(csv_path='..\\..\\hw4_data\\train.csv',data_dir='..\\..\\hw4_data\\train')
    train_sampler = CategoriesSampler( n_batch, train_classes , train_way, shot + query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=0, pin_memory=False)
    
    valset = MiniDataset(csv_path='..\\..\\hw4_data\\val.csv',data_dir='..\\..\\hw4_data\\val')
    val_sampler =CategoriesSampler( n_batch, test_classes , test_way, shot + query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=0, pin_memory=False)
    
    model = Convnet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)



    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, (data, _) in enumerate(train_loader, 1):
            # print("i=",i,"batch[0]=",batch[0],"batch[1]=",batch[1])
            data = data.to(device)
            
            p = shot * train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(shot, train_way, -1).mean(dim=0)

            label = torch.arange(train_way).repeat(query)
            label = label.long().to(device)

            logits = cosine_similarity(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)


            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        tl = tl.item()
        ta = ta.item()
        
        # tensorboard
        writer.add_scalar('acc_mean/train', ta, epoch)
        writer.add_scalar('loss/train', tl, epoch)
        
        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, tl, ta))
        
        model.eval()

        vl = Averager()
        va = Averager()

        for i, (data, _) in enumerate(val_loader, 1):
            data = data.to(device)

            p = shot * test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(shot, test_way, -1).mean(dim=0)

            label = torch.arange(test_way).repeat(query)
            label = label.long().to(device)

            logits = cosine_similarity(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        # tensorboard
        writer.add_scalar('acc_mean/validation', va, epoch)
        writer.add_scalar('loss/validation', vl, epoch)



        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(output_path, 'trlog'))

        save_model('epoch-last')

        if epoch % save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / max_epoch)))
        

    
    writer.close()
