import argparse
from pathlib import Path
from configs.config import return_defaults
from utils.data import ExhaustiveWalksDataset, ExhaustiveWalksDB






def main():
    parser = argparse.ArgumentParser(description='GraphGL Network')
    parser.add_argument('--point', default=(51.509865, -0.118092))
    parser.add_argument('--width', default=2000)
    parser.add_argument('--workers', default=4)
    args = vars(parser.parse_args())
    dictlist = []
    for key, value in list(args.items()): dictlist.append(key), dictlist.append(value) 

    cfg = return_defaults()
    cfg.merge_from_list(dictlist)
    cfg.freeze()
    path = Path(cfg.path) / 'data' / 'graphs'
    # data = ExhaustiveWalksDataset(root=path, args=cfg)
    # path = Path(cfg.path) / 'data' / 'graphs'
    data = ExhaustiveWalksDB(root=path, args=cfg)

if __name__ == '__main__':
    main()