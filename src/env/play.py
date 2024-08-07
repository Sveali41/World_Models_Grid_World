from itertools import count
import sys
sys.path.append('/home/siyao/project/rlPractice/dlai_project')
from src.env.rollout_generator import RolloutGenerator
import hydra
from omegaconf import DictConfig, OmegaConf
from src.common.utils import PROJECT_ROOT, get_env
import torch 
from collections import Counter
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="/home/siyao/project/rlPractice/dlai_project/conf/hparams", config_name="config")
def main(cfg: DictConfig):
    # cfg.test_env.visualize = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rg = RolloutGenerator(cfg,device)
    cumulative_reward = 0
    count_rewards=Counter()
    n_rollouts = cfg.test_env.n_rollouts
    for _ in range(n_rollouts):
        rew = - rg.rollout(params=None)
        count_rewards.update([rew])
        cumulative_reward += rew
    print("Reward after {} rollouts: {}".format(n_rollouts, cumulative_reward))
    print("Avg Reward {}".format(cumulative_reward/n_rollouts))

    plt.figure(figsize=(10,10))
    _ = plt.bar(count_rewards.keys(),count_rewards.values()) 
    plt.title("Bar Plot of reward frequency") 
    plt.show()
    print(count_rewards)
    log_dir = get_env('LOG_FOLDER')
    plt.savefig(log_dir+'/play_reward_bar_plot.svg')

    

if __name__ == "__main__":
    main()