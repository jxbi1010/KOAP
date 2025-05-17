import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

OBS_HORIZON = 2

class IDM_Agent(nn.Module):

    def __init__(self,
                 latent_policy: DictConfig,
                 idm_baseline: DictConfig,
                 trainset: DictConfig,
                 obs_encoder:DictConfig,
                 visual_input: bool = False,
                 scale_data: bool = False,
                 data_ratio:int = 100,
                 device: str = 'cuda'):

        super(IDM_Agent, self).__init__()

        self.device = device
        self.visual_input = visual_input
        self.latent_policy =  hydra.utils.instantiate(latent_policy)
        self.idm_baseline = hydra.utils.instantiate(idm_baseline)

        self.obs_encoder = hydra.utils.instantiate(obs_encoder)
        self.trainset = hydra.utils.instantiate(trainset)
        # self.scaler = Scaler(self.trainset.get_all_observations(), self.trainset.get_all_actions(),scale_data=False,device = device)
        # self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        # self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        print('init idm policy')

    def load_model_from_ckpt(self,latent_policy_ckpt_path, idm_baseline_ckpt_path):

        self.latent_policy.load_pretrained_model(latent_policy_ckpt_path,"eval_best_idm.pth")
        self.idm_baseline.load_pretrained_model(idm_baseline_ckpt_path,"eval_best_idm.pth")

        print(f'load idm_baseline from {idm_baseline_ckpt_path}')


    @torch.no_grad()
    def predict(self, inputs, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:

        self.latent_policy.model.eval()
        self.idm_baseline.model.eval()

        inputs = torch.from_numpy(inputs).float().to(self.device).unsqueeze(0)

        pred_obs = self.latent_policy.predict(inputs)
        full_obs = torch.cat((inputs,pred_obs),dim=1)
        pred_action = self.idm_baseline.predict(full_obs)

        return pred_action


    def reset(self):
        pass
