import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('/home/siyao/project/rlPractice/dlai_project')
from src.common.utils import PROJECT_ROOT,get_env
from src.data.datamodule import WMRLDataModule
from src.models.mdnrnn import MDNRNN
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
# reproducibility stuff
import numpy as np
import random
import torch
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
torch.backends.cudnn.benchmark = False
_ = pl.seed_everything(0)

wandb.require("core")
@hydra.main(version_base=None, config_path=PROJECT_ROOT / "conf/hparams", config_name="config")
def train(cfg: DictConfig):
    hparams = cfg
    dataloader = WMRLDataModule(hparams = hparams.mdnrnn)
    # Instantiate the model
    mdnrnn = MDNRNN(hparams=hparams)
    # Define the logger
    # https://www.wandb.com/articles/pytorch-lightning-with-weights-biases.
    wandb_logger = WandbLogger(project="MDNRNN WM", log_model=True)
    # ## Currently it does not log the model weights, there is a bug in wandb and/or lightning.
    wandb_logger.experiment.watch(mdnrnn, log='all', log_freq=1000)
    # Define the trainer
    metric_to_monitor = 'avg_val_loss_mdnrnn'#"loss"
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, patience=15, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
                            save_top_k=1,
                            monitor = metric_to_monitor,
                            mode = "min",
                            dirpath = get_env('PTH_FOLDER'),
                            filename ="mdnrnn-{epoch:02d}-{avg_val_loss_mdnrnn:.4f}",
                            verbose = True
                        )
    trainer = pl.Trainer(logger=wandb_logger,
                    max_epochs=hparams.mdnrnn.n_epochs, 
                    gpus=1,
                    callbacks=[early_stop_callback, checkpoint_callback])     
    # Start the training
    trainer.fit(mdnrnn,dataloader)
    # Log the trained model
    model_pth = hparams.mdnrnn.pth_folder
    trainer.save_checkpoint(model_pth)
    wandb.save(str(model_pth))

if __name__ == "__main__":
    train()