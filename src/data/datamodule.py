import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.common.utils import get_env
from typing import Tuple, List, Any, Dict, Optional
import os.path
# import src.env.run_env_save as env_run_save
import numpy as np
import torch
import multiprocessing
import time
from func_timeout import func_set_timeout

class WMRLDataset(Dataset):
    @func_set_timeout(100)
    def __init__(self,loaded ,hparams):
        self.hparams = hparams
        self.data = self.make_data(loaded)

    @func_set_timeout(100)
    def make_data(self,loaded):
        obs = loaded['a']
        act = loaded['b']
        transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),#https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
                    ])
        data = []
        len = obs.shape[0]
        # length of the sequence ls is a variable for readability, +1 because 
        # for the training item we want to split in [0:seq] [1:seq+1]
        ls = self.hparams.seq_len+1 
        if ls > 2:
            done = loaded['d'].astype(int) # to convert boolean values in binary 0-1
            # if the size is greater than 2 we construct a "sequence" dataset
            for idx in range(len-ls):
                obs_t = torch.stack([transform(obs[i]) for i in range(idx,idx+ls)])
                # we only need ls-1 actions, because the last pred is needed for prediction
                act_t = torch.tensor([[act[i]] for i in range(idx,idx+ls-1)]) 
                # we want to predict the terminal state after taking the action in the current state
                done_t = torch.tensor([[done[i]] for i in range(idx+1,idx+ls)], dtype=torch.float32) #
                data.append({"obs":obs_t, "act":act_t, "done":done_t})
        else:
            for idx in range(len):
                obs_t = transform(obs[idx])
                act_t = torch.tensor(act[idx])
                data.append({"obs":obs_t,"act":act_t})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class WMRLDataModule(pl.LightningDataModule):
    def __init__(self, hparams = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_dir = self.hparams.data_dir
        print(self.hparams)
        
    def setup(self, stage: Optional[str] = None):
        loaded = np.load(self.data_dir)
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # # Map the load_file function to the list of files
        #     loaded = pool.map(np.load(self.data_dir))
        data = WMRLDataset(loaded,self.hparams)
        split_size=int(len(data)*9/10)
        self.data_train, self.data_test = torch.utils.data.random_split(data, \
                                        [split_size, len(data)-split_size])

    def train_dataloader(self):
        return DataLoader(
                self.data_train, 
                batch_size = self.hparams.batch_size, 
                shuffle = True,
                num_workers = self.hparams.n_cpu,
                pin_memory=True,
                persistent_workers=True
            )

    def val_dataloader(self):
       return DataLoader(
                self.data_test, 
                batch_size = self.hparams.batch_size, 
                shuffle = False,
                num_workers = self.hparams.n_cpu,
                pin_memory=True,
                persistent_workers=True
            )

