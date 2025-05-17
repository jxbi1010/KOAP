import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
import hydra
from typing import Optional


log = logging.getLogger(__name__)

OBS_HORIZON = 2

class IDM_Agent(nn.Module):

    def __init__(self,
                 latent_policy: DictConfig,
                 idm_model: DictConfig,
                 device: str = 'cuda'):

        super(IDM_Agent, self).__init__()

        self.device = device
        self.latent_policy =  hydra.utils.instantiate(latent_policy)
        self.idm_model = hydra.utils.instantiate(idm_model)

        print('init idm policy')

    def load_model_from_ckpt(self,latent_policy_ckpt_path, action_decoder_ckpt_path):

        self.latent_policy.load_pretrained_model(latent_policy_ckpt_path,"eval_best_idm.pth")
        print(f'action_decoder_ckpt_path:{action_decoder_ckpt_path}')
        self.idm_model.load_pretrained_model(action_decoder_ckpt_path,"eval_best_idm.pth")
        print(f'load idm_model from {action_decoder_ckpt_path}')


    @torch.no_grad()
    def predict(self, inputs, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:

        self.latent_policy.model.eval()
        self.idm_model.model.eval()

        inputs = torch.from_numpy(inputs).float().to(self.device).unsqueeze(0)

        pred_obs = self.latent_policy.predict(inputs)
        full_obs = torch.cat((inputs,pred_obs),dim=1)
        pred_action = self.idm_model.predict(full_obs)

        return pred_action


    def reset(self):
        pass
