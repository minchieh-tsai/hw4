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
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            Image.open,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
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

def euclidean_metric(a, b):
    n_quary = a.shape[0]
    m_mean = b.shape[0]
    a = a.unsqueeze(1).expand(n_quary, m_mean, -1)
    b = b.unsqueeze(0).expand(n_quary, m_mean, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def add_noise(a, b):
    n_support = a.shape[0]
    M_rate = b.shape[0]
    a = a.unsqueeze(0).expand(M_rate, n_support, -1)
    b = b.unsqueeze(1).expand(M_rate, n_support, -1)
    noised = (a + b).view(M_rate * n_support, -1)
    return noised

def predict(args, model, hallu, data_loader, M_rate, DEVICE):
    

    pred = []
    model.eval()
    hallu.eval()
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            print(i)
            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:].to(DEVICE)
            query_input   = data[args.N_way * args.N_shot:,:,:,:].to(DEVICE)
            
            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.tensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])
            query_label = query_label.long().to(DEVICE)
            
            
            output = model(support_input)   # (shot * train_way, 1600)
            
            # generate fake output
            noise = torch.randn(M_rate, 1600, device=DEVICE)
            noised = add_noise(output, noise)   # (M_rate * shot * train_way, 1600)
            fake_output = hallu(noised)
            
            # calculate prototype
            output = output.reshape(args.N_shot, args.N_way, -1)
            fake_output = fake_output.reshape(args.N_shot * M_rate, args.N_way, -1)
            proto = torch.cat([output,fake_output], 0).mean(dim=0)
            
            # TODO: classify the query data depending on the its distense with each prototype
            logits = euclidean_metric(model(query_input), proto)
            predict = torch.argmax(logits, dim=1)
            
            pred.append(predict.tolist())
            
            # if i > 5:
            #     break
    
    return pred

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)
 
    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))
    
    
    M = 50
    M_rate = int(M/(args.N_way * args.N_shot))
    
    # TODO: load your model
    # DEVICE = torch.device("cuda:1")
    DEVICE = torch.device('cpu')
    checkpoint = torch.load(args.load , map_location = 'cpu')
    
    model_ft = Convnet()
    model_ft.load_state_dict(checkpoint["cnn"])
    model_ft = model_ft.to(DEVICE)
    
    hallu = Hallucinator()
    hallu.load_state_dict(checkpoint["hallu"])
    hallu = hallu.to(DEVICE)
    
    prediction_results = predict(args, model_ft, hallu, test_loader, M_rate, DEVICE)
    
    
    # # TODO: output your prediction to csv
    df = pd.DataFrame(prediction_results, 
                      columns =[f"query{i}" for i in range(75)]
                      ).rename_axis('episode_id')
    
    df.to_csv(os.path.join(args.output_csv), index=1)
    
    
    
    
    
    
    
    
    