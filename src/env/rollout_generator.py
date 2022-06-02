from pickle import FALSE
from src.models.vae import VAE
from src.models.mdnrnn import MDNRNN
from src.models.controller import CONTROLLER
import torch
from torchvision import transforms
import gym
from gym_minigrid.wrappers import *
from os.path import exists
import time
#taken from the original implementation 
def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
#taken from the original implementation
def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened
#taken from the original implementation
def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    def __init__(self, hparams, device):
        """ Build vae, rnn, controller and environment. """
        self.hparams = hparams
        self.device = device
        self.time_limit = hparams.test_env.time_limit
        vae_pth = hparams.vae.pth_folder
        mdnrnn_pth = hparams.mdnrnn.pth_folder
        controller_pth = hparams.controller.pth_folder
        assert exists(vae_pth) and exists(mdnrnn_pth), \
            "Either vae or mdrnn is untrained."
        self.vae = VAE.load_from_checkpoint(vae_pth).to(self.device).to(self.device)
        self.mdnrnn = MDNRNN.load_from_checkpoint(mdnrnn_pth, strict = False).to(self.device)
        self.controller = CONTROLLER(hparams.controller).to(self.device)
        if exists(controller_pth):
            self.controller.load_state_dict(torch.load(controller_pth, map_location=self.device)['state_dict'])
        self.env = gym.make(hparams.test_env.env_name)#, n_obstacles=2 to change obstacles in the obstacles world
        #self.env.env.width, self.env.env.height = 8,8 # to work with different size env
        # self.env = StateBonus(self.env)
        self.env = RGBImgObsWrapper(self.env) # Get pixel observations
        self.env = ImgObsWrapper(self.env) # Get rid of the 'mission' field

    def get_action_and_transition(self, obs, hidden):
        _, _, _, z = self.vae(obs)
        action = self.controller.choose_an_action([z, hidden[0][0]]).unsqueeze(0)
        (_, _, _, done), next_hidden = self.mdnrnn(z.unsqueeze(0), action.to(self.device), hidden = hidden)
        return action.squeeze().cpu().numpy(), next_hidden, done.view(-1).detach().cpu().numpy()[0]

    def rollout(self, params):
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)
        obs = self.env.reset()
        hidden = [torch.zeros(self.hparams.mdnrnn.num_layers, 1, self.hparams.mdnrnn.hidden_size).to(self.device) for _ in range(2)] # 1 always to have "fake batch dim"
        transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),#https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
                    ])
        cumulative = 0
        i = 0
        while True:
            if self.hparams.test_env.visualize:
                self.env.render()
                time.sleep(0.1)
            obs = transform(obs).unsqueeze(0).to(self.device) # we need to make it "batch" to work with pytorch models
            action, hidden, pdone = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)
            cumulative += reward #- pdone
            if done or i > self.time_limit:
                if i > self.time_limit:
                    cumulative-=1
                # I add a special reward that depends on the steps we survive 
                return - cumulative#- i/(self.time_limit*10)
            i += 1
