from loader import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pycuda.driver as cuda
cuda.init()
from models import TwoLayerNet
from torch.utils.tensorboard import SummaryWriter
import time


class trainer:
    def __init__(self, model, lossfn, optimizer, device, tb):
        self.model = model
        self.lossfn = lossfn
        self.optimizer = optimizer
        self.device = device
        self.writer = tb

    def eval(self, X, Y):
        preds = self.model(X)
        loss = self.lossfn(preds, Y)

        return loss.item()

    def splitData(self, dataloader, test_idx, val_idx):
        
        return (val, test, train)

    def fit(self, model_params, dataset, epochs, k=10, nbatches=10):
        batchsize = int(len(dataset)/nbatches)
        dataloader = DataLoader(dataset, batch_size = batchsize, shuffle = False,
                                drop_last = True)




        for nfold in range(nbatches):
            print(nfold)
            val_idx = nfold
            test_idx = int((nfold + 1) % nbatches)
            idx = list(range(nbatches))

            val_x, val_y = [], []
            test_x, test_y = [] , []

            net = self.model(*model_params)
            net.to(self.device)

            criterion = self.lossfn()
            optimizer = self.optimizer(net.parameters(), lr = 0.001)

            print(next(net.parameters()).is_cuda)

            for i, data in enumerate(dataloader):
                if i == val_idx:
                    val_x, val_y = data[0], data[1]
                elif i == test_idx:
                    test_x, test_y = data[0], data[1]
                else:
                    continue

            best_loss = float("inf")
            best_model = net.state_dict()
            start_t = time.time()
            for epoch in range(epochs):
                running_loss = 0
                net.train()
                for i, data in enumerate(dataloader):
                    if i == test_idx or i == val_idx:
                        continue
                    inputs, labels = data[0], data[1]
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    val_preds = net(val_x)
                    val_loss = criterion(val_preds, val_y)
                    best_model = net.state_dict()
                    #self.writer.add_scalar('validation loss', val_loss.item(), epoch)
                    #self.writer.add_scalar('training loss', running_loss/nbatches, epoch)

            net.load_state_dict(best_model)
            net.eval()
            test_preds = net(test_x)
            test_loss = criterion(test_preds, test_y)
            print("Test loss for fold {}: {}".format(nfold, test_loss.item()))
            print("time diff: ", time.time() - start_t)


if __name__ == '__main__':
    writer = SummaryWriter('runs/metasense_experiment_1')
    device = 'cuda'
    metasense = MetaSenseDataset(device)
    modelTrainer = trainer(TwoLayerNet, nn.L1Loss, optim.Adam, device, writer)
    modelTrainer.fit((6, 200, 1), metasense, 50)

