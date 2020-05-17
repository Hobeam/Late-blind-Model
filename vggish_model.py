import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import vggish_params as params

import pdb

class Vggish(nn.Module):
    def __init__(self):
        super(Vggish, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
        #     nn.Conv2d(2, 64, kernel_size=3, padding=1),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU()
        #     )
        self.features = self.make_layers()

        # self.fc = nn.Sequential(
        #     nn.Linear(512*6*4, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, 100),
        #     nn.BatchNorm1d(100, affine=False)
        #     )

        self.fc = nn.Sequential(
            nn.Linear(512*6*4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 100),
            nn.BatchNorm1d(100, affine=False),
            nn.Dropout()
            )

    def make_layers(self):
        layers = []
        in_channels = 1
        for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        # pdb.set_trace()
        x = x.view(batch_size, -1)
        # pdb.set_trace()
        x = self.fc(x)
        return x

class MyModel(nn.Module):
    def __init__(self, num_class=10, num_unit=1024, weights_path=None):
        super(MyModel, self).__init__()

        self.vggish = Vggish()
        self.classifer = nn.Sequential(
            nn.Linear(100, num_unit),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(num_unit, num_class)
            )

        if weights_path is not None:
            self.load_weights(weights_path)

    def forward(self, x):
        x = self.vggish(x)
        x = self.classifer(x)
        # pdb.set_trace()
        return F.softmax(x, dim=1)

    def load_weights(self, weights_path):
        data = np.load(weights_path)
        weights = data['dict'][()]

        weights_name = weights.keys()

        for name, param in self.named_parameters():
            if name in weights_name and 'vggish' in name:
                # print name
                param.data = torch.from_numpy(weights[name])
