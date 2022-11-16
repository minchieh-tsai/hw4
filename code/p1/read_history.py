# -*- coding: utf-8 -*-

import os
import torch
import matplotlib.pyplot as plt

if __name__=='__main__':
    models = ["20201224_2122", "cos_smiliary_20201225_2341"]
    parameters = ["train_loss", "val_loss", "train_acc", "val_acc"]
    checkpoints = {}
    
    for model in models:
        log_path = os.path.join("log", model, "trlog")
        checkpoint = torch.load(log_path , map_location = 'cpu')
        checkpoints[model] = checkpoint
        print(model)
        print("max_acc=", checkpoint["max_acc"])
        
    for parameter in parameters:
        plt.figure()
        for model in models:
            plt.plot(checkpoints[model][parameter])
            plt.title(parameter)
            




