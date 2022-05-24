import hydra
from omegaconf import DictConfig, OmegaConf
from src.common.utils import PROJECT_ROOT
from src.env.data.datamodule import WMRLDataModule

@hydra.main(version_base=None, config_path=PROJECT_ROOT / "conf/hparams", config_name="config")
def train(cfg: DictConfig):
    hparams = cfg
    dataloader = WMRLDataModule(hparams = hparams)
    dataloader.setup()
    batch = iter(dataloader.train_dataloader()).next()
    print(batch)
    print(batch['obs'].dtype)


if __name__ == "__main__":
    train()