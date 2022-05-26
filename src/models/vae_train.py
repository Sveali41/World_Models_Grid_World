import hydra
from omegaconf import DictConfig, OmegaConf
from src.common.utils import PROJECT_ROOT
from src.env.data.datamodule import WMRLDataModule
from src.models.vae2 import VAE
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
wandb.require("service")
@hydra.main(version_base=None, config_path=PROJECT_ROOT / "conf/hparams", config_name="config")
def train(cfg: DictConfig):
    hparams = cfg.vae
    dataloader = WMRLDataModule(hparams = hparams)
    # Instantiate the model
    vae = VAE(hparams=hparams)
    # Define the logger
    # https://www.wandb.com/articles/pytorch-lightning-with-weights-biases.
    wandb_logger = WandbLogger(project="VAE WM", log_model=True)
    # ## Currently it does not log the model weights, there is a bug in wandb and/or lightning.
    wandb_logger.experiment.watch(vae, log='all', log_freq=1000)
    # Define the trainer
    trainer = pl.Trainer(logger=wandb_logger,
                        max_epochs=hparams.n_epochs, 
                        gpus=1)    

    # Start the training
    trainer.fit(vae,dataloader)
    # Log the trained model
    model_pth = PROJECT_ROOT/"src/models/pth/vae.pth"
    trainer.save_checkpoint(model_pth)
    wandb.save(str(model_pth))

if __name__ == "__main__":
    train()