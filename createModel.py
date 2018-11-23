import torch.nn as nn
from torchvision import models
from ConvLSTMCell import *

class ViolenceModel(nn.Module):
    def __init__(self,modelused,pretrained):
        super(ViolenceModel, self).__init__()
        if modelused=='alexnet':
                self.mod=models.alexnet(pretrained=pretrained)
                self.convNet = nn.Sequential(*(list(self.mod.children()))[:-1])
                self.mem_size=256
                self.conv_lstm = ConvLSTMCell(256, self.mem_size)
        elif modelused=='resnet50':
                self.mod=models.resnet50(pretrained=pretrained)
                self.convNet = nn.Sequential(*(list(self.mod.children()))[:-2])
                self.mem_size=512
                self.conv_lstm = ConvLSTMCell(2048, self.mem_size)
        elif modelused=='vgg16':
                self.mod=models.vgg16(pretrained=pretrained)
                self.convNet = nn.Sequential(*(list(self.mod.children()))[:-1])
                self.mem_size=256
                self.conv_lstm = ConvLSTMCell(512, self.mem_size)
        elif modelused=='vgg19':
                self.mod=models.vgg19(pretrained=pretrained)
                self.convNet = nn.Sequential(*(list(self.mod.children()))[:-1])
                self.mem_size=384
                self.conv_lstm = ConvLSTMCell(384, self.mem_size)
        elif modelused=='resnet18':
                self.mod=models.resnet18(pretrained=pretrained)
                self.convNet = nn.Sequential(*(list(self.mod.children()))[:-2])
                self.mem_size=768
                self.conv_lstm = ConvLSTMCell(768, self.mem_size)

        elif modelused=='resnet101':
                self.mod=models.resnet101(pretrained=pretrained)
                self.convNet = nn.Sequential(*(list(self.mod.children()))[:-2])
                self.mem_size=1536
                self.conv_lstm = ConvLSTMCell(1536, self.mem_size)

        if pretrained:
            for param in self.convNet.parameters():
                param.requires_grad = False
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(3*3*self.mem_size, 1000)
        self.lin2 = nn.Linear(1000, 256)
        self.lin3 = nn.Linear(256, 32)
        self.lin4 = nn.Linear(32, 4)
        self.BN = nn.BatchNorm1d(1000)
        self.classifier = nn.Sequential(self.lin1, self.BN, self.relu, self.lin2, self.relu,
                                        self.lin3, self.relu, self.lin4)

    def forward(self, x):
        state = None
        seqLen = x.size(0) - 1
        for t in range(0, seqLen):
            x1 = x[t] - x[t+1]
            x1 = self.convNet(x1)
            state = self.conv_lstm(x1, state)
        x = self.maxpool(state[0])
        x = self.classifier(x.view(x.size(0), -1))
        return x