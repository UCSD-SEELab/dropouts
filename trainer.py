from test import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pycuda.driver as cuda
cuda.init()
from models import TwoLayerNet


class trainer:
    def __init__(self, model, lossfn, optimizer, device):
        self.model = model
        self.lossfn = lossfn
        self.optimizer = optimizer
        self.device = device

    def eval(self, X, Y):
        preds = self.model(X)
        loss = self.lossfn(preds, Y)

        return loss.item()

    def fit(self, dataset, epochs, k=10):
        dataloader = DataLoader(dataset, batch_size = int(len(dataset)/10),
                                shuffle = False, drop_last = True)

        for 

