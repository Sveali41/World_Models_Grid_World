import gym3
from procgen import ProcgenGym3Env
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import sys
from src.common.utils import PROJECT_ROOT

@hydra.main(version_base=None, config_path=PROJECT_ROOT / "conf", config_name="maze/config")
def run_env(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    env = ProcgenGym3Env(
                        env_name="maze", # Name of environment, or comma-separate list of environment names to instantiate as each env in the VecEnv.
                        num=15, # The number of unique levels that can be generated. Set to 0 to use unlimited levels.
                        start_level=0, # The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
                        distribution_mode="easy",# All games support "easy" and "hard", while other options are game-specific. The default is "hard". Switching to "easy" will reduce the number of timesteps required to solve each game and is useful for testing or when working with limited compute resources.
                        render_mode="rgb_array", 
                    )
    env = gym3.ViewerWrapper(env, info_key="rgb")
    step = 0
    while True:
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        print(f"step {step} reward {rew} first {first}")
        step += 1

if __name__ == "__main__":
    run_env()
