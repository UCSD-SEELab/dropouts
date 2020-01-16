import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import metaLoader as readings

class MetaSenseDataset(Dataset):
    def __init__(self, dev='cpu'):
        self.data = readings.load()
        self.dev = torch.device(dev)
        self.tensor_x = torch.tensor(self.data[:,:-1], dtype=torch.float).to(self.dev)
        self.tensor_y = torch.tensor(self.data[:,-1].reshape(len(self.data), 1),
                                     dtype = torch.float).to(self.dev)
        #self.tensor_x = torch.stack([torch.tensor(datum, dtype=torch.float)
        #                             for datum in self.data[:,:-1]]).to(self.dev)
        #self.tensor_y = torch.stack([torch.tensor(datum, dtype=torch.float)
        #                             for datum in self.data[:,-1].reshape((len(self.data), 1))]).to(self.dev)

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.tensor_x[idx], self.tensor_y[idx]

        return datum


if __name__ == '__main__':
    device = 'cuda'
    metasense = MetaSenseDataset(device)
    dataloader = DataLoader(metasense, batch_size = int(len(metasense)/10),
                            shuffle = True, drop_last = True)

    for t in dataloader:
        pass

