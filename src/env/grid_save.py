import gym3
from procgen import ProcgenGym3Env
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from src.common.utils import PROJECT_ROOT
import gym
from gym_minigrid.wrappers import *
import random
def run_env(cfg: DictConfig):
   
    env = gym.make("MiniGrid-Dynamic-Obstacles-8x8-v0") #'MiniGrid-Empty-8x8-v0'
    env = RGBImgObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    # if (cfg.env.visualize): #set to false to hide the gui
    #     env = env.render 
    obs_list, act_list, rew_list, done_list = [], [], [], []
    episodes=0
    env.reset()
    while episodes < cfg.collect.episodes:
        act = random.randint(0, env.action_space.n - 1)
        obs, reward, done, _ = env.step(act)
        act_list.append([act])
        obs_list.append([obs])
        rew_list.append([reward])
        done_list.append([done])
        if done:
            episodes += 1
            print(episodes)
            env.reset()
        env.render('rgb_array')
    obs_np = np.concatenate(obs_list)
    act_np = np.concatenate(act_list)
    rew_np = np.concatenate(rew_list)
    done_np = np.concatenate(done_list)
    print(obs_np.shape)
    print(act_np.shape)
    print(rew_np.shape)
    print(done_np.shape)
    print("Num episodes started: ", episodes)
    return obs_np, act_np, rew_np, done_np

def save_experiments(cfg: DictConfig, obs, act, rew, done):
    np.savez_compressed(cfg.collect.data_train, a=obs, b=act, c=rew, d=done)

@hydra.main(version_base=None, config_path = PROJECT_ROOT / "conf/env", config_name="config")
def main(cfg: DictConfig):
    obs, act,rew, done = run_env(cfg)
    save_experiments(cfg,obs,act,rew, done)

if __name__ == "__main__":
    main()
