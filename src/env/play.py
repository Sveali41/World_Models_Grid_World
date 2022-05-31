from src.env.rollout_generator import RolloutGenerator
import hydra
from omegaconf import DictConfig, OmegaConf
from src.common.utils import PROJECT_ROOT, get_env
import torch 

@hydra.main(version_base=None, config_path=PROJECT_ROOT / "conf/hparams", config_name="config")
def main(cfg: DictConfig):
    cfg.test_env.visualize = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rg = RolloutGenerator(cfg,device)
    while True:
        rew = rg.rollout(params=None)
        print("reward: ",rew)

if __name__ == "__main__":
    main()