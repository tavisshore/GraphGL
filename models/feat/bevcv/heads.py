from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ProjectionModule(pl.LightningModule):
    def __init__(self, scale=1):
        super(ProjectionModule, self).__init__()
        self.pov_proj = nn.Sequential(
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        )
        self.map_proj = nn.Sequential(
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        )
        self.scale = scale

    def forward(self, pov=None, map=None):
        if pov is not None: pov = self.pov_proj(pov)
        if map is not None: map = self.map_proj(map)
        return pov, map


class ShapeStandardiser(pl.LightningModule):
    def     __init__(self):
        super(ShapeStandardiser, self).__init__()
        self.mp = nn.Sequential(
            nn.MaxPool2d(kernel_size=(7, 7), stride=None, padding=0),
            nn.Flatten(1, -1))
        self.reduce = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU())
        self.scale = 1

    def standard_shape(self, x):
        shape = x.shape
        if len(shape) > 2:
            if shape[2] == 7: x = self.mp(x)
        if shape[1] == 2048: x = self.reduce(x)

        x = x.view(-1, 512)
        x_in = F.normalize(x, p=2, dim=1) * self.scale
        return x_in
    
    def forward(self, pov=None, map=None):
        if pov == None and map==None:
            raise ValueError('Both pov and map cannot be None')
        if pov == None:
            map_out = self.standard_shape(map)
            return map_out
        if map == None:
            pov_out = self.standard_shape(pov)
            return pov_out
        if pov != None and map != None:
            pov_out = self.standard_shape(pov)
            map_out = self.standard_shape(map)
            return pov_out, map_out
