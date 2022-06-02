import torch
import torch.nn as nn
import pytorch_lightning as pl
# easiest model structure (just a mlp) but hardest to train since it has to solve a non-linear, non-convex continuous optimization problem.
# CMA-ES will be used 
class CONTROLLER(pl.LightningModule):
    """ Controller """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.fc = nn.Linear(self.hparams.latent_size + self.hparams.hidden_size, self.hparams.action_size)

    def forward(self, inputs): #input is a list [latents, recurrents]
        cat_in = torch.cat(inputs, dim=1)
        # print(cat_in.shape)
        return torch.tanh(self.fc(cat_in))
    def choose_an_action(self, inputs):
        # there are 2 actions: turn right and go forward. This space has been divided
        # into a continuous action space between -1.0 to 1.0, and divided this range into half
        continuous_action = self(inputs)
        if not self.hparams.discrete_action_space:
            return continuous_action
        discrete_action = torch.ones_like(continuous_action)
        n_actions = self.hparams.n_actions
        # for i in range(1,n_actions):


        discrete_action[continuous_action<0]=2
        return discrete_action