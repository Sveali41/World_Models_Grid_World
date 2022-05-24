import gym3
from procgen import ProcgenGym3Env
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from src.common.utils import PROJECT_ROOT
import random

def run_env(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    env = ProcgenGym3Env(
                        env_name = cfg.env.env_name, # Name of environment, or comma-separate list of environment names to instantiate as each env in the VecEnv.
                        num = 1, # number of parallel env to generate each time
                        num_levels = cfg.env.num_levels, # The number of unique levels that can be generated. Set to 0 to use unlimited levels.
                        start_level = cfg.env.start_level, # The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
                        distribution_mode = cfg.env.distribution_mode,# All games support "easy" and "hard", while other options are game-specific. The default is "hard". Switching to "easy" will reduce the number of timesteps required to solve each game and is useful for testing or when working with limited compute resources.
                        render_mode = "rgb_array", #to visualize it
                    )

    print("action space")
    print(env.ac_space) # action space
    print("combos")
    print(env.combos) # all the available combinations of actions under form of combos

    if (cfg.env.visualize): #set to false to hide the gui
        env = gym3.ViewerWrapper(env, info_key="rgb") 

    obs_list, act_list = [], []
    episodes=0
    from_first = 0
    for step in range(cfg.collect.steps):
        from_first+=1
        act = gym3.types_np.sample(env.ac_space, bshape=(env.num,))
        env.act(act)
        rew, obs, first = env.observe()
        #to have a more solid dataset: we save only 10% of the first steps
        if from_first < 50:
            rand = random.uniform(0,1)
            if rand > 0.1:
                continue
        act_list.append(act)
        obs_list.append(obs['rgb'])
        if first[0]:
            print(f"step {step} reward {rew} first {first}")
            from_first=0
            episodes+=1
    obs_np = np.concatenate(obs_list)
    act_np = np.concatenate(act_list)
    print(obs_np.shape)
    print(act_np.shape)
    print("Num episodes started: ", episodes)
    return obs_np, act_np

def save_experiments(cfg: DictConfig, obs, act):
    np.savez_compressed(cfg.collect.data_train, a=obs, b=act)

@hydra.main(version_base=None, config_path = PROJECT_ROOT / "conf/env", config_name="config")
def main(cfg: DictConfig):
    obs, act = run_env(cfg)
    save_experiments(cfg,obs,act)

if __name__ == "__main__":
    main()
