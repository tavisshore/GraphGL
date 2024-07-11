import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Optional, Tuple, Union
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torchvision.transforms import v2
import torchvision
from torch_geometric.nn.models import GraphSAGE, GAT # inductive
from torch.nn import functional as F

from pytorch_metric_learning import losses

from models.feat.bevcv.bevcv import BEVCV


from utils.data import GraphData, GraphDataset 
from utils.metric import recall_accuracy

EPS = 1e-15

def barlow_twins_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim
    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)
    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size
    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = ((1 - c.diagonal()).pow(2).sum() + _lambda * c[off_diagonal_mask].pow(2).sum())
    return loss


def mask_features(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    x = x + torch.randn_like(x) * p
    return x


def drop_edges(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    num_edges = edge_index.size(-1)
    device = edge_index.device
    mask = bernoulli_mask(size=num_edges, prob=p) == 1.
    return edge_index[:, mask]

def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return torch.bernoulli((1 - prob) * torch.ones(size))


class GraphAugmentor:
    def __init__(self, p_x_1: float, p_e_1: float, p_x_2: Optional[float] = None, p_e_2: Optional[float] = None, query: str = 'sat'):
        self._p_x_1 = p_x_1
        self._p_e_1 = p_e_1
        self._p_x_2 = p_x_2 if p_x_2 is not None else p_x_1
        self._p_e_2 = p_e_2 if p_e_2 is not None else p_e_1
        self.query = query

    def __call__(self, data: Data, stage='train'):
        if stage == 'train':
            edge_index_a = drop_edges(data.edge_index, p=self._p_e_1)
            edge_index_b = drop_edges(data.edge_index, p=self._p_e_2)
        else:
            edge_index_a = data.edge_index
            edge_index_b = data.edge_index
        x_a = mask_features(data.x, p=self._p_x_1)
        if self.query == 'sat': x_b = mask_features(data.x, p=self._p_x_1)
        elif self.query == 'pov': x_b = mask_features(data.pov, p=self._p_x_2)
        return (x_a, edge_index_a), (x_b, edge_index_b)


class FullModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Feature Extractor
        self.feat_extractor = BEVCV(config=self.args)
        if self.args.feat_ext_pretrained: 
            self.feat_extractor.load_state_dict(torch.load(f'{self.args.path}/weights/bevcv_90.pt', map_location=self.device), strict=False)
            print(f'Loaded BEV-CV Weights')
        if self.args.feat_ext_freeze: self.feat_extractor.eval()

        if self.args.encoder == 'gat':    self.encoder = GAT(in_channels=512, hidden_channels=self.args.hidden_dim, num_layers=self.args.enc_layers, out_channels=self.args.out_dim)
        elif self.args.encoder == 'sage': self.encoder = GraphSAGE(in_channels=512, hidden_channels=self.args.hidden_dim, num_layers=self.args.enc_layers, out_channels=self.args.out_dim)

        # self.train_augmentor = GraphAugmentor(p_x_1=self.args.p_x_1, p_e_1=self.args.p_e_1, p_x_2=self.args.p_x_2, p_e_2=self.args.p_e_2)
        # self.val_augmentor = GraphAugmentor(p_x_1=self.args.eval.p_x_1, p_e_1=self.args.eval.p_e_1, p_x_2=self.args.eval.p_x_2, p_e_2=self.args.eval.p_e_2)
        # REMOVE
        torchvision.disable_beta_transforms_warning()
        self.augmentor = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.1),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.val_process = v2.Compose([
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

        self.loss_function = losses.TripletMarginLoss(margin=0.1) # NTXent?
        
        self.embs, self.btchs = [], []
        self.batch_size = self.args.batch_size
        self.current_val_loss = 100
        self.prepare_data()

    def sat_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.feat_extractor.embed_pov(pov_tile=x)
        x = self.forward(x=x, edge_index=edge_index)
        return x
    
    def pov_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.feat_extractor.embed_map(map_tile=x)
        x = self.forward(x=x, edge_index=edge_index)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x=x, edge_index=edge_index)
        return F.normalize(x, p=2, dim=1)
    
    def prepare_data(self):
        data = GraphData(self.args)
        self.train_dataset = GraphDataset(self.args, data, stage='train')
        self.val_dataset = GraphDataset(self.args, data, stage='val')

    def train_dataloader(self): return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

    def val_dataloader(self): return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    def step_function(self, batch, stage='train'): 
        x_sat, ei_sat = batch['sat'], batch['edge_index']
        x_pov, ei_pov = batch['pov'], batch['edge_index']
        if stage == 'train': x_sat, x_pov = self.augmentor(x_sat), self.augmentor(x_pov)
        else: x_sat, x_pov = self.val_process(x_sat), self.val_process(x_pov)
        z_a = self.sat_forward(x=x_sat, edge_index=ei_sat)
        z_b = self.pov_forward(x=x_pov, edge_index=ei_pov)
        if self.args.loss == 'triplet':
            labels = torch.arange(z_a.size(0), device=z_a.device)
            negative_labels = torch.tensor([torch.where(labels != label)[0][torch.abs(torch.where(labels != label)[0] - label) > 
                                                                            self.args.neg_distance][0] for label in labels], device=z_a.device)
            loss = self.loss_function(embeddings=z_a, ref_emb=z_b, indices_tuple=(labels, labels, negative_labels))
        else:
            loss = barlow_twins_loss(z_a=z_a, z_b=z_b)
        self.log(f'{stage}_loss', loss.item(), batch_size=self.batch_size, sync_dist=True)
        return {'loss': loss, 'z_a': z_a, 'z_b': z_b, 'batch': batch}
    
    def training_step(self, batch, batch_idx): return self.step_function(batch, stage='train')

    def validation_step(self, batch, batch_idx): return self.step_function(batch, stage='val')

    def epoch_function(self, outputs, stage='val'):
        losses, embs, batch = [], [], []
        for output in outputs: losses.append(output['loss']), embs.append((output['z_a'], output['z_b'])), batch.append(output['batch'])

        epoch_loss = sum(losses)/len(losses)
        self.log(f'{stage}_epoch_loss', epoch_loss, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            top_1, top_5, top_10, top_1_per = recall_accuracy(args=self.args, data=batch, embeddings=embs)
            self.log(f'{stage}_top_1', top_1, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{stage}_top_5', top_5, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{stage}_top_10', top_10, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{stage}_top_1%', top_1_per, prog_bar=True, logger=True, sync_dist=True)

    def training_epoch_end(self, outputs): self.epoch_function(outputs=outputs, stage='train')

    def validation_epoch_end(self, outputs): self.epoch_function(outputs=outputs, stage='val')

    def configure_optimizers(self): 
        opt = torch.optim.AdamW(params=self.encoder.parameters(), lr=0.001, weight_decay=1e-5)
        sch = ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.1, patience=6, verbose=True)
        return {'optimizer': opt, 'lr_scheduler': sch, 'monitor': 'train_epoch_loss'}


        


