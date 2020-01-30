from test import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pycuda.driver as cuda
import time
cuda.init()

class OneLayerNet(nn.Module):
    def __init__(self, d_in, H, d_out, drop_p=0):
        super(OneLayerNet, self).__init__()
        self.l1 = nn.Linear(d_in, H)
        self.l2 = nn.Linear(H, H)
        self.d1 = nn.Dropout(p=drop_p)
        self.l3 = nn.Linear(H, d_out)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))

        return out

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, H, d_out, drop_p=0):
        super(TwoLayerNet, self).__init__()
        self.l1 = nn.Linear(d_in, H)
        self.l2 = nn.Linear(H, H)
        self.d1 = nn.Dropout(p=drop_p)
        self.l3 = nn.Linear(H, H)
        self.d2 = nn.Dropout(p=drop_p)
        self.l4 = nn.Linear(H, d_out)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.l4(out))

        return out

class ThreeLayerNet(nn.Module):
    def __init__(self, d_in, H, d_out, drop_p=0):
        super(ThreeLayerNet, self).__init__()
        self.l1 = nn.Linear(d_in, H)
        self.l2 = nn.Linear(H, H)
        self.d1 = nn.Dropout(p=drop_p)
        self.l3 = nn.Linear(H, H)
        self.d2 = nn.Dropout(p=drop_p)
        self.l4 = nn.Linear(H, H)
        self.d3 = nn.Dropout(p=drop_p)
        self.l5 = nn.Linear(H, d_out)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.d1(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.d2(out))
        out = F.relu(self.l4(out))
        out = F.relu(self.d3(out))
        out = F.relu(self.l5(out))

        return out



if __name__ == '__main__':
    device = 'cuda'
    metasense = MetaSenseDataset(device)
    dataloader = DataLoader(metasense, batch_size = int(len(metasense)/20),
                            shuffle = False, drop_last = True)

    net = TwoLayerNet(6, 200, 1)
    net.to(device)
    print(next(net.parameters()).is_cuda)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True)


    for epoch in range(50):
        running_loss = 0.0
        ctr = 0
        running_loss = 0
        start_t = time.time()
        for i, data in enumerate(dataloader):
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("time: %s" %(time.time() - start_t))
        print("Loss at epoch {}: {}".format(epoch, running_loss/20))




