import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.common.utils import get_env
from typing import Tuple, List, Any, Dict, Optional
import os.path
import src.env.run_env_save as env_run_save
import numpy as np
import torch


class WMRLDataset(Dataset):
    def __init__(self, obs,act,hparams):
        self.data = self.make_data(obs,act)
        self.hparams = hparams
        
    def make_data(self,obs,act):
        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),#https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
                        ])
        data = []
        for idx in range(obs.shape[0]):
            obs_t = transform(obs[idx])
            act_t = torch.tensor(act[idx])
            data.append({"obs":obs_t,"act":act_t})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class WMRLDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = get_env("TRAIN_DATASET_PATH")+"/train_dataset.npz", hparams = None):
        super().__init__()
        self.data_dir = data_dir
        self.save_hyperparameters(hparams)
        print(self.hparams)
    def setup(self, stage: Optional[str] = None):
        loaded = np.load(self.data_dir)
        obs = loaded['a']
        act = loaded['b']
        data = WMRLDataset(obs,act,self.hparams)
        size_split = int(len(data)*10/11)
        self.data_train = data[:size_split]
        self.data_test = data[size_split:]

    def train_dataloader(self):
        return DataLoader(
                self.data_train, 
                batch_size = self.hparams.batch_size, 
                shuffle = True,
                num_workers = self.hparams.n_cpu,
                pin_memory=True
            )

    def val_dataloader(self):
       return DataLoader(
                self.data_test, 
                batch_size = self.hparams.batch_size, 
                shuffle = False,
                num_workers = self.hparams.n_cpu,
                pin_memory=True
            )

