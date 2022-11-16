import numpy as np
import os
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_1 import MiniDataset
from sampler import CategoriesSampler
from model import Convnet, Hallucinator

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

def add_noise(a, b):
    n_support = a.shape[0]
    M_rate = b.shape[0]
    a = a.unsqueeze(0).expand(M_rate, n_support, -1)
    b = b.unsqueeze(1).expand(M_rate, n_support, -1)
    noised = (a + b).view(M_rate * n_support, -1)
    return noised

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
    
if __name__ == '__main__':
    
    t = time.localtime()
    result = time.strftime("%Y%m%d_%H%M", t)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    M               = 100
    M_rate_train    = int(M/(train_way * shot))
    M_rate_test     = int(M/(test_way * shot))
    output_path = f'log/M{M}_{result}'

    
    writer = SummaryWriter(output_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    trainset = MiniDataset(csv_path='..\\..\\hw4_data\\train.csv',data_dir='..\\..\\hw4_data\\train')
    train_sampler = CategoriesSampler(n_batch, train_classes , train_way, shot + query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=0, pin_memory=False)
    
    valset = MiniDataset(csv_path='..\\..\\hw4_data\\val.csv',data_dir='..\\..\\hw4_data\\val')
    val_sampler =CategoriesSampler( n_batch, test_classes , test_way, shot + query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=0, pin_memory=False)
    
    model = Convnet().to(device)
    hallu = Hallucinator().to(device)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(hallu.parameters()), lr=0.001)
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
            
            output = model(data_shot)   # (shot * train_way, 1600)
            
            # generate fake output
            noise = torch.randn(M_rate_train, 1600, device=device)
            noised = add_noise(output, noise)   # (M_rate * shot * train_way, 1600)
            fake_output = hallu(noised)
            
            # calculate prototype
            output = output.reshape(shot, train_way, -1)
            fake_output = fake_output.reshape(shot * M_rate_train, train_way, -1)
            proto = torch.cat([output,fake_output], 0).mean(dim=0)
            
            label = torch.arange(train_way).repeat(query)
            label = label.long().to(device)
            
            logits = euclidean_metric(model(data_query), proto)
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
            
            output = model(data_shot)
            
            # generate fake output
            noise = torch.randn(M_rate_test, 1600, device=device)
            noised = add_noise(output, noise)   # (M_rate * shot * test_way, 1600)
            fake_output = hallu(noised)
            
            # calculate prototype
            output = output.reshape(shot, test_way, -1)
            fake_output = fake_output.reshape(shot * M_rate_test, test_way, -1)
            proto = torch.cat([output,fake_output], 0).mean(dim=0)
            
            
            label = torch.arange(test_way).repeat(query)
            label = label.long().to(device)

            logits = euclidean_metric(model(data_query), proto)
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
            torch.save({'cnn'   :model.state_dict(), 
                        'hallu' :hallu.state_dict(), 
                       }, osp.join(output_path, 'max-acc' + '.pth'))
        
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)
        torch.save(trlog, osp.join(output_path, 'trlog'))
        
        if epoch % save_epoch == 0:
            torch.save({'cnn'   :model.state_dict(), 
                        'hallu' :hallu.state_dict(), 
                       }, osp.join(output_path, f'epoch-{epoch}' + '.pth'))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / max_epoch)))
        
        
    
    writer.close()
