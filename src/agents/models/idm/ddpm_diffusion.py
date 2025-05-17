# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from agents.models.idm.util import unsqueeze_xdim


def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1 ** 2 + sigma2 ** 2
    coef1 = sigma2 ** 2 / denom
    coef2 = sigma1 ** 2 / denom
    var = (sigma1 ** 2 * sigma2 ** 2) / denom
    return coef1, coef2, var


class DDPMDiffusion():
    def __init__(self, opt):
        device = opt.device
        self.device = device

        betas = self.make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval // 2], np.flip(betas[:opt.interval // 2])])

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

        self.noise_scheduler = DDPMScheduler(
        num_train_timesteps=opt.interval,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, ot_ode=False):
        noise = torch.randn(x0.shape, device=self.device)

        noisy_actions = self.noise_scheduler.add_noise(
            x0, noise, step)

        return noisy_actions, noise

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n ** 2 - std_nprev ** 2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(self, x1, ema, net, cond=None, diffuse_step=None):

        with ema.average_parameters(net.parameters()):
            diffuse_steps = self.noise_scheduler.timesteps if diffuse_step is None else self.noise_scheduler.timesteps[:int(diffuse_step)]
            # init scheduler
            for k in diffuse_steps:
                # predict noise
                noise_pred = net(
                    sample=x1,
                    timestep=k,
                    global_cond=cond
                )

                # inverse diffusion step (remove noise)
                x1 = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=x1
                ).prev_sample

        return x1

    def make_beta_schedule(self, n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
        # return np.linspace(linear_start, linear_end, n_timestep)
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
        return betas.numpy()

