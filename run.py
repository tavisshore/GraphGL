from pathlib import Path
import argparse
import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as plg

from configs.config import return_defaults
from models.model import FullModel

from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
class CustomRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

progress_bar = CustomRichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="#e55151",
        processing_speed="#ff9738",
        metrics="#FFFFFF",
    ),
    leave=True
)

def main():
    parser = argparse.ArgumentParser(description='GraphGL Network')

    cfg = return_defaults()
    list_of_args = dict(cfg).keys()
    for k in list_of_args:
        if k != 'eval':
            parser.add_argument('--config', default='initial/single_single') if k == 'config' else parser.add_argument(f'--{k}')
        else: 
            for i in dict(cfg[k]).keys(): parser.add_argument(f'-{i}')
    args = vars(parser.parse_args())
    dictlist = []
    for key, value in list(args.items()): 
        if key in ['single_walk', 'single_node', 'p_x_1', 'p_e_1', 'p_x_2', 'p_e_2']: 
            if value is not None: dictlist.append(f'eval.{k}'), dictlist.append(value) 
        else: 
            if value is not None: dictlist.append(key), dictlist.append(value) 

    cfg.merge_from_file(f'{cfg.path}/configs/{args["config"]}.yaml')
    cfg.merge_from_list(dictlist)
    cfg.freeze()

    ckpt_dir = Path(f'{cfg.path}/weights/checkpoints/{args["config"]}')
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    t = datetime.datetime.now()
    time_string = "_".join(str(t.time()).split(":")[:3])
    pl.seed_everything(42)

    name = f'{cfg.name}_{time_string}'

    wandb_logger = plg.WandbLogger(entity="bev-cv-project", project="GraphGL", save_dir=f'{cfg.path}/logs/wandb/', name=name, version=name, log_model=False)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_epoch_loss", mode="min", dirpath=ckpt_dir, save_top_k=1, filename=name)
    wandb_logger.log_hyperparams(dict(cfg)) 

    torch.set_float32_matmul_precision('high')

    model = FullModel(args=cfg)
    trainer = pl.Trainer(max_epochs=cfg.epochs, accelerator='gpu', 
                        logger=[wandb_logger], 
                        callbacks=[checkpoint_callback, 
                                    progress_bar
                                    ],
                        check_val_every_n_epoch=cfg.val_interval, num_sanity_val_steps=0,
                        strategy='ddp'
                        )
    trainer.fit(model)

    trainer.test(model)



if __name__ == '__main__':
    main()