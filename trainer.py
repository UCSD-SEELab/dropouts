from loader import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pycuda.driver as cuda
cuda.init()
from models import TwoLayerNet
from torch.utils.tensorboard import SummaryWriter
import time
import pandas as pd


class trainer:
    def __init__(self, model, lossfn, optimizer, device, tb, label):
        self.model = model
        self.lossfn = lossfn
        self.optimizer = optimizer
        self.device = device
        self.writer = tb
        self.label = label

    def eval(self, X, Y):
        preds = self.model(X)
        loss = self.lossfn(preds, Y)

        return loss.item()

    def splitData(self, dataloader, test_idx, val_idx):
        
        return (val, test, train)

    def fit(self, model_params, dataset, epochs, k=10, nbatches=10, DEBUG=False):
        batchsize = int(len(dataset)/nbatches)
        dataloader = DataLoader(dataset, batch_size = batchsize, shuffle = False,
                                drop_last = True)
        self.data_table = []
        final_test_loss = []

        for nfold in range(nbatches):
            val_idx = nfold
            test_idx = int((nfold + 1) % nbatches)
            idx = list(range(nbatches))

            val_x, val_y = [], []
            test_x, test_y = [] , []
            fold_train_data = [self.label, "train", nfold]
            fold_val_data = [self.label, "validation", nfold]
            fold_test_data = [self.label, "test", nfold]

            net = self.model(*model_params)
            net.to(self.device)

            criterion = self.lossfn()
            optimizer = self.optimizer(net.parameters(), lr = 0.001)

            if DEBUG:
                print("fold : ", nfold)
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

            if DEBUG:
                start_t = time.time()

            for epoch in range(epochs):
                running_loss = 0
                net.train()
                for i, data in enumerate(dataloader):
                    if (i == test_idx) or (i == val_idx):
                        continue
                    inputs, labels = data[0], data[1]
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                running_loss /= nbatches
                fold_train_data.append(running_loss)

                if DEBUG:
                    print("fold {} training loss: {}".format(nfold, running_loss))

                with torch.no_grad():
                    val_preds = net(val_x)
                    val_loss = criterion(val_preds, val_y).item()
                    fold_val_data.append(val_loss)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = net.state_dict()
                    #self.writer.add_scalar('validation loss', val_loss.item(), epoch)
                    #self.writer.add_scalar('training loss', running_loss/nbatches, epoch)


            net.load_state_dict(best_model)
            net.eval()
            test_preds = net(test_x)
            test_loss = criterion(test_preds, test_y).item()
            print("Test loss for fold {}: {}".format(nfold, test_loss))
            fold_test_data.append(test_loss)
            final_test_loss.append(test_loss)
            self.data_table.append(fold_train_data)
            self.data_table.append(fold_val_data)
            self.data_table.append(fold_test_data)

        cols = ['label', 'data', 'fold'] + list(range(epochs))
        self.data_table = pd.DataFrame(self.data_table)
        print("Final loss on test data: ", np.mean(final_test_loss))

    def logger(self):
        fname = self.label + '_readings.csv'
        self.data_table.to_csv(fname, index=False)

if __name__ == '__main__':
    writer = SummaryWriter('runs/metasense_experiment_1')
    device = 'cuda'
    metasense = MetaSenseDataset(device)
    
    ########### One Layer ############
    print("------------------ One Layer Net ------------------")
    modelTrainer = trainer(OneLayerNet, nn.L1Loss, optim.Adam, device, writer, "1-layer")
    modelTrainer.fit((6, 200, 1), metasense, 50)
    modelTrainer.logger()
    print("===================================================")

    ########### Two Layer ############
    print("------------------ Two Layer Net ------------------")
    modelTrainer = trainer(TwoLayerNet, nn.L1Loss, optim.Adam, device, writer, "2-layer")
    modelTrainer.fit((6, 200, 1), metasense, 50)
    modelTrainer.logger()
    print("===================================================")

    ########### Three Layer ############
    print("------------------ Three Layer Net ------------------")
    modelTrainer = trainer(ThreeLayerNet, nn.L1Loss, optim.Adam, device, writer, "2-layer")
    modelTrainer.fit((6, 200, 1), metasense, 50)
    modelTrainer.logger()
    print("===================================================")


