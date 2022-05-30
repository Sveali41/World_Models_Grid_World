from asyncio.log import logger
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from src.common.utils import PROJECT_ROOT
from src.models.vae import VAE
import numpy as np
import wandb
import math
import torchvision.utils
from typing import Sequence, List, Dict, Tuple, Optional, Any, Set, Union, Callable, Mapping
# code based on https://github.com/ctallec/world-models/blob/master/models/mdrnn.py and https://github.com/arnaudvl/world-models-ppo/blob/master/mdnrnn/mdnrnn.py
# https://github.com/hardmaru/WorldModelsExperiments/blob/c0cb2dee69f4b05d9494bc0263eca25a7f90d555/carracing/rnn/rnn.py#L139
# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
# https://github.com/AppliedDataSciencePartners/WorldModels/blob/master/rnn/arch.py#L39
# https://github.com/JunhongXu/world-models-pytorch
# https://github.com/sksq96/pytorch-mdn/blob/master/mdn-rnn.ipynb
# Recurrent Mixture Density Network.
class MDNRNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if hasattr(hparams,'vae'):
            #if true we'll need it to train
            self.vae=VAE.load_from_checkpoint(hparams.vae.pth_folder)
            self.vae.freeze() #we do not want to train it
            hparams = hparams.mdnrnn
        self.save_hyperparameters(hparams)
        self.z_size = hparams.latent_size
        self.n_hidden = hparams.hidden_size
        self.n_gaussians = hparams.n_gaussians
        self.n_layers = self.hparams.num_layers
        self.action_size = self.hparams.action_size
        self.lstm = nn.LSTM(self.z_size+self.action_size, self.n_hidden, self.n_layers, batch_first=True)
        # I could have implemented only one linear layer here, to modularize better the network I decided to split it.
        self.fc1 = nn.Linear(self.n_hidden, self.n_gaussians*self.z_size) # to compute pi
        self.fc2 = nn.Linear(self.n_hidden, self.n_gaussians*self.z_size) # to compute mu
        self.fc3 = nn.Linear(self.n_hidden, self.n_gaussians*self.z_size) # to compute sigma
        #this is a classification problems, to predict terminate state (0, 1)
        self.fc4 = nn.Linear(self.n_hidden, 1)
    
    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y) 
        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.z_size)
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma) #to have positive values
        return pi, mu, sigma
    
    def forward(self, latents, actions, hidden = None):
        x = torch.cat([actions, latents], dim=-1)
        # Forward propagate LSTM
        if hidden == None:
            y, (h, c) = self.lstm(x)
        else:
            y, (h, c) = self.lstm(x, hidden)
        pi, mu, sigma = self.get_mixture_coef(y)
        done = torch.sigmoid(self.fc4(y)) # done is a binary value, sigmoid is ideal here.
        return (pi, mu, sigma, done), (h, c)

    def mdn_loss_fn(self,y, mu, sigma, pi):
        # Actually, the loss is not lower bounded and the problem is actually ill-posed, 
        # since one of the mixture components may collapse in a data point, making the 
        # loss decrease to arbitrarily small values (i.e. going to −∞).
        # https://stats.stackexchange.com/questions/294567/negative-loss-while-training-gaussian-mixture-density-networks
        # many problems also in the original implementation, they saw that nothing change training or not the network.
        # check also the other references above, all have issues in this negative value, this is a huge cons of this method.
        y = y.unsqueeze(2)
        m = Normal(loc=mu, scale=sigma)
        loss = torch.exp(m.log_prob(y))
        loss = torch.sum(loss * pi, dim=2) #sum on gaussians dimension
        loss = -torch.log(loss)
        return loss.mean()
    
    def done_loss(self, predict, original):
        predict = predict.view(-1, predict.shape[-1])
        original = original.view(-1, original.shape[-1])
        return F.mse_loss(predict, original, reduction='mean')

    def loss_function(self, next_latent_obs, mu, sigma, pi, pdone, done):
        MDNL = self.mdn_loss_fn(next_latent_obs, mu, sigma, pi)
        DONEL = self.done_loss(pdone, done)
        logger_d = {'loss':MDNL+DONEL,'mdn_loss': MDNL, 'done_loss':DONEL}
        return logger_d
    
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        #RMSprop is recommended for RNNs
        # optimizer = optim.RMSprop(params, lr=self.hparams.lr,alpha=.9)
        optimizer = optim.Adam(params, lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }

    # helper training function
    def img2latent(self,obs):
        """ Function to go from image to latent space. """
        original_shape = obs.shape
        obs = obs.reshape(-1,*original_shape[2:])
        with torch.no_grad():
            _, mu, logsigma, z = self.vae(obs)
            latent = (mu + logsigma.exp() * torch.randn_like(mu)).view(*original_shape[0:2], self.hparams.latent_size)
        return z.view(*original_shape[0:2], self.hparams.latent_size)

    def training_step(self, batch, batch_idx):
        obs = batch['obs']
        act = batch['act']
        done = batch['done']
        obs_pre = obs[:,0:self.hparams.seq_len, ...]
        obs_next = obs[:,1:,...]
        latent_obs = self.img2latent(obs_pre)
        # print(latent_obs.shape)
        next_latent_obs = self.img2latent(obs_next)
        (pi, mu, sigma, pdone), (_,_) = self(latent_obs,act)
        loss = self.loss_function(next_latent_obs, mu, sigma, pi, pdone, done)
        self.log_dict(loss)
        return loss['loss']

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]:
        obs = batch['obs']
        act = batch['act']
        done = batch['done']
        obs_pre = obs[:,0:self.hparams.seq_len, ...]
        obs_next = obs[:,1:,...]
        latent_obs = self.img2latent(obs_pre)
        next_latent_obs = self.img2latent(obs_next)
        (pi, mu, sigma, pdone), (_,_) = self(latent_obs,act)
        loss = self.loss_function(next_latent_obs, mu, sigma, pi, pdone, done)
        return {"loss_mdnrnn_val": loss['loss']}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]]]:
        avg_loss = torch.stack([x["loss_mdnrnn_val"] for x in outputs]).mean()
        self.log_dict({"avg_val_loss_mdnrnn": avg_loss})
        return {"avg_val_loss_mdnrnn": avg_loss}
    
    def on_save_checkpoint(self,checkpoint):
        # pop the backbone here using custom logic
        t = checkpoint['state_dict']
        checkpoint['state_dict'] =  {key: t[key] for key in t if not key.startswith('vae.')}
