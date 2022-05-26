import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import torchvision.utils
import pytorch_lightning as pl
from typing import Sequence, List, Dict, Tuple, Optional, Any, Set, Union, Callable, Mapping
# code based on https://github.com/ctallec/world-model, https://github.com/Deepest-Project/WorldModels-A3C
# and notebooks shown in class

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super(Encoder, self).__init__()
        # flatten_dims = hp.img_height//2**4
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 32, 48, 48)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 64, 24, 24)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 128, 12, 12)
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 256, 6, 6)
        )
        self.fc = nn.Linear(6*6*256, latent_dims*2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1) # (B, d)
        h = self.fc(h) # (B, )
        mu = h[:, :self.latent_dims]
        logvar = h[:, self.latent_dims:]
        # sigma = self.softplus(h[:, self.latent_dims:])
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dims, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 6, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 128, 6, 6)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 64, 12, 12)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 32, 24, 24)
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 32, 48, 48)
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            # nn.Tanh()
            nn.Sigmoid()
            # nn.LeakyReLU(), # (B, c, 96, 96)
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), -1, 2, 2)
        y = self.decoder(h)
        return y   

class VAE(pl.LightningModule):
    """ Variational Autoencoder """
    def __init__(self, hparams):
        super(VAE, self).__init__()
        self.save_hyperparameters(hparams)
        self.encoder = Encoder(self.hparams.img_channels, self.hparams.latent_size)
        self.decoder = Decoder(self.hparams.img_channels, self.hparams.latent_size)
        # It avoids wandb logging when lighting does a sanity check on the validation
        self.is_sanity = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        x_recon = self.decoder(z)
        return x_recon, mu, logsigma

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }

    def loss_function(self,recon_x, x, mu, logsigma):
        # (from notebook) You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
        # another reference https://arxiv.org/abs/1312.6114
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
        return {"loss": BCE + KLD, "BCE": BCE, "KLD": KLD}

    def training_step(self, batch, batch_idx):
        obs = batch['obs']
        recon_batch, mu, logvar = self(obs)
        loss = self.loss_function(recon_batch, obs, mu, logvar)
        self.log_dict(loss)
        return loss['loss']

    # from cyclegan notebook
    def get_image_examples(self, real: torch.Tensor, reconstructed: torch.Tensor) -> Sequence[wandb.Image]:
        """
        Given real and "fake" translated images, produce a nice coupled images to log
        :param real: the real images
        :param reconstructed: the reconstructed image

        :returns: a sequence of wandb.Image to log and visualize the performance
        """
        example_images = []
        for i in range(real.shape[0]):
            couple = torchvision.utils.make_grid(
                [real[i], reconstructed[i]],
                nrow=2,
                normalize=True,
                scale_each=False,
                pad_value=1,
                padding=4,
            )
            example_images.append(
                wandb.Image(couple.permute(1, 2, 0).detach().cpu().numpy(), mode="RGB")# no need of .permute(1, 2, 0) since pil image
            )
        return example_images

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]:
        obs = batch['obs']
        recon_batch, mu, logvar = self(obs)
        loss = self.loss_function(recon_batch, obs, mu, logvar)
        images = self.get_image_examples(obs, recon_batch)
        return {"loss_vae_val": loss['loss'], "images": images}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]]]:
        """ Implements the behaviouir at the end of a validation epoch

        Currently it gathers all the produced examples and log them to wandb,
        limiting the logged examples to `hparams["log_images"]`.

        Then computes the mean of the losses and returns it. 
        Updates the progress bar label with this loss.

        :param outputs: a sequence that aggregates all the outputs of the validation steps

        :returns: the aggregated validation loss and information to update the progress bar
        """
        images = []

        for x in outputs:
            images.extend(x["images"])
            
        images = images[: self.hparams.log_images]

        if not self.is_sanity:  # ignore if it not a real validation epoch. The first one is not.
            print(f"Logged {len(images)} images for each category.")
            self.logger.experiment.log(
                {f"images": images},
                step=self.global_step,
            )
        self.is_sanity = False

        avg_loss = torch.stack([x["loss_vae_val"] for x in outputs]).mean()
        self.log_dict({"avg_val_loss_vae": avg_loss})
        return {"avg_val_loss_vae": avg_loss}
