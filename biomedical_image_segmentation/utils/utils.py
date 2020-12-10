import random
import numpy as np

def get_ids(): #might be useful later on
    """Returns a list of the ids in the directory"""
    return (f for f in os.listdir('dataset/'))

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    #return {'train': dataset[:1], 'val': dataset[:1]}
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

