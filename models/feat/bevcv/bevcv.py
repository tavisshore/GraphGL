import torch
from torch import nn
import torchvision.transforms as T

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from models.feat.bevcv.bev import BEVNET
from models.feat.bevcv.heads import ProjectionModule, ShapeStandardiser



class BEVCV(pl.LightningModule):
    def __init__(self, config=None):
        super(BEVCV, self).__init__()
        self.save_hyperparameters()
        self.cfg = config
        self.current_val_loss = 10
        self.calib_demo = torch.Tensor([[630, 0.0, 694.4383], [0.0, 630, 241.6793], [0.0, 0.0, 1.0]])

        bev_model = BEVNET()
        self.pov_fpn = bev_model.frontend
        self.pov_transformer = bev_model.transformer
        self.bev_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=3), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=3), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1), nn.BatchNorm2d(512), nn.LeakyReLU())
        self.map_unet = smp.Unet(encoder_name="resnet50", in_channels=3, classes=2, activation='sigmoid')
        self.map_branch = self.map_unet.encoder

        self.standard_shaper = ShapeStandardiser()
        self.projection = ProjectionModule(scale=1)
        # self.normalise = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def embed_map(self, map_tile):
        # map_tile = self.normalise(map_tile)
        map_embedding = self.map_branch(map_tile)[5]
        map_embedding = self.standard_shaper(None, map_embedding)
        map_embedding = self.projection(map_embedding)
        return map_embedding
    
    def embed_pov(self, pov_tile, calib=None):
        # pov_tile = self.normalise(pov_tile)
        feature_maps = self.pov_fpn(pov_tile)
        if calib is None: calib = self.calib_demo.repeat(pov_tile.shape[0], 1, 1).cuda()
        bev_feats = self.pov_transformer(feature_maps, calib)
        pov_embedding = self.bev_conv(bev_feats)
        pov_embedding = self.standard_shaper(pov_embedding, None)
        pov_embedding = self.projection(pov_embedding)
        return pov_embedding







