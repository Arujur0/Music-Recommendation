import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlk(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding = 0):
        super(ConvBlk, self).__init__()
        self.convblk = nn.Sequential(nn.Conv2d(nin, nout, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(nout), nn.ReLU())
    def forward(self, x):
        x = self.convblk(x)
        return x
    
class Emotional(nn.Module):
    def __init__(self, nin, noc):
        super(Emotional, self).__init__()
        self.blk1 = ConvBlk(nin, 128, 3)
        self.blk2 = ConvBlk(128, 128, 3, padding = 'same')
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)

        self.blk3 = ConvBlk(128, 64, 3, padding = 'same')
        self.blk4 = ConvBlk(64, 64, 3, padding = 'same')

        self.blk5 = ConvBlk(64, 32, 3, padding = 'same')
        self.blk6 = ConvBlk(32, 32, 3, padding = 'same')

        self.flat = nn.Flatten()
        self.Lin = nn.Sequential(nn.Linear(800, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.Lin2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.Lin3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU())

        self.Classifier = nn.Linear(128,noc)

    
    def forward(self, x):
        
        x = self.max(self.blk2(self.blk1(x)))
        
        x1 = self.max(self.blk4(self.blk3(x)))
        
        x2 = self.max(self.blk6(self.blk5(x1)))
        
        x3 = self.Lin3(self.Lin2(self.Lin(self.flat(x2))))
        
        results = self.Classifier(x3)
        
        return results