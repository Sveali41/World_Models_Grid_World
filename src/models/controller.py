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
        # there are 3 actions: turn right, turn left and go forward. This space has been divided
        # into a continuous action space between -1.0 to 1.0, and divided this range into thirds
        continuous_action = self(inputs).view(-1)
        if not self.hparams.discrete_action_space:
            return continuous_action
        interval1 = -1+0.666666
        interval2 = interval1+0.666666
        if continuous_action <= interval1:
            action = 0
        elif continuous_action <= interval2:
            action = 1
        else:
            action = 2
        return torch.tensor([action]).unsqueeze(0).unsqueeze(1)