"""
A slightly modification on the customized VGG13 in:
    Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution, 2016
    Emad Barsoum, Cha Zhang, Cristian Canton Ferrer and Zhengyou Zhang
"""

from collections import OrderedDict

import torch
from torch import nn
from torchvision import transforms


DEFAULT_WEIGHT_PATH = 'weight/best.ckpt'


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class VGG13(nn.Module):
    def __init__(
        self, input_channel=1, fc=1024, avg_pool=2,
        class_=7, cnn_dropout=.25, fc_dropout=.6
    ):
        super(VGG13, self).__init__()

        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU(True)),
            ('pool', nn.MaxPool2d(2)),
            ('dropout', nn.Dropout(cnn_dropout))
        ]))

        self.block2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU(True)),
            ('pool', nn.MaxPool2d(2)),
            ('dropout', nn.Dropout(cnn_dropout))
        ]))

        self.block3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU(True)),
            ('conv3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(True)),
            ('pool', nn.MaxPool2d(2)),
            ('dropout', nn.Dropout(cnn_dropout))
        ]))

        self.block4 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU(True)),
            ('conv3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(True)),
            ('pool', nn.MaxPool2d(2)),
            ('dropout', nn.Dropout(cnn_dropout))
        ]))

        self.avg_pool = nn.AdaptiveAvgPool2d(avg_pool)
        self.flatten = Flatten()

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(256 * avg_pool ** 2, fc)),
            ('relu1', nn.ReLU(True)),
            ('dropout1', nn.Dropout(fc_dropout)),
            ('fc2', nn.Linear(fc, fc)),
            ('relu1', nn.ReLU(True)),
            ('dropout2', nn.Dropout(fc_dropout))
        ]))

        self.classifier = nn.Linear(fc, class_)

    def forward(self, x):
        return self.classifier(self.features(x))

    def features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    @classmethod
    def from_state_dict(cls, ckpt_path, gpu=False):
        if gpu:
            checkpoint = torch.load(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

        model = cls()
        model.load_state_dict(checkpoint)
        return model


def get_model(_input_image_size=96):
    return VGG13()


def get_preprocess(input_image_size=96):
    return transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(input_image_size),
    transforms.ToTensor(),
])
