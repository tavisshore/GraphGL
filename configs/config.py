from yacs.config import CfgNode as CN
from pathlib import Path

path = str(Path(__file__).parent.parent.absolute())

_C = CN()
# defaults 
_C.data = 'data'
_C.dataroot = 'data/images/'
_C.savedir = 'data'
_C.lr = 1e-4

_C.batch_size = 12
_C.weight_decay = 1e-5
_C.epochs = 250
_C.checkpoint_dir = 'weights/checkpoints/'
_C.point = (51.509865, -0.118092)#(51.243594, -0.576837)
_C.width = 4000
_C.embedding = 'bevcv'
_C.crop_size = 128
_C.embedder = 'simple'
_C.single_val = False
_C.frozen_weights = None
_C.walk = 3

_C.workers=4

_C.config = 'initial/single_single'
_C.resume_training = False
_C.path = path
_C.fov = 90

_C.loss = 'triplet'
_C.encoder = 'gat'
_C.enc_layers = 2
_C.hidden_dim = 256
_C.out_dim = 64
_C.query = 'sat' # sat or pov

_C.neg_distance = 10
_C.name = 'test'
_C.val_interval = 2

# evaluation
_C.eval = CN()
_C.eval.single_walk = False # one node embedding for each node in corpus
_C.eval.single_node = False # That node embedding is the start_point only

_C.p_x_1 = 0.0
_C.p_e_1 = 0.0
_C.p_x_2 = 0.0
_C.p_e_2 = 0.0
_C.eval.p_x_1 = 0.0
_C.eval.p_e_1 = 0.0
_C.eval.p_x_2 = 0.0
_C.eval.p_e_2 = 0.0

### Cofiguration about bev-cv pretrained or not
_C.feat_ext_pretrained = True
_C.feat_ext_freeze = False
# _C.end2end = True

_C.train_split = 0.8

def return_defaults():
    return _C.clone()

