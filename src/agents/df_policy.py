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

from agents.base_agent import BaseAgent
from agents.utils.scaler import Normalizer
from agents.models.idm.util import build_optimizer_sched

from agents.models.idm.ddpm_diffusion import DDPMDiffusion
from agents.models.idm.conditional_unet_1D import DiffusionConditionalUnet1D
# from torch_ema import ExponentialMovingAverage
# from agents.models.diffusion.ema import ExponentialMovingAverage
from agents.models.idm.ema import ExponentialMovingAverage
log = logging.getLogger(__name__)

OBS_HORIZON = 2


class DF_Policy(nn.Module):

    def __init__(self,
                 diffusion_opt: DictConfig,
                 input_dim,
                 act_dim,
                 visual_input: bool = False,
                 device: str = 'cuda'):
        super(DF_Policy, self).__init__()

        self.opt = diffusion_opt
        self.input_dim = input_dim
        self.act_dim = act_dim

        self.diffusion = DDPMDiffusion(self.opt)
        self.latent_policy = DiffusionConditionalUnet1D(input_dim=act_dim,
                                                        global_cond_dim=OBS_HORIZON * 10).to(device)

        self.ema = ExponentialMovingAverage(self.latent_policy.parameters(), decay=self.opt.ema)

    def forward(self, obs, act):

        # make prediction
        obs_cond = obs[:, 0:OBS_HORIZON]
        act_future = act[:, OBS_HORIZON:OBS_HORIZON + 12]

        B = obs.shape[0]

        # diffusion latent policy
        timesteps = torch.randint(
            0, self.diffusion.noise_scheduler.config.num_train_timesteps,
            (B,), device=obs.device
        ).long()
        # sample noise to add to actions
        noisy_z, noise = self.diffusion.q_sample(timesteps, act_future)

        # predict the noise residual
        noise_pred = self.latent_policy(
            noisy_z, timesteps, global_cond=obs_cond.flatten(start_dim=1))

        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def sample(self, past_cond,act):
        self.eval()

        # diffusion latent policy
        noisy_act = torch.randn((act.size(0), 12, act.shape[-1]), device=act.device)

        pred_act = self.diffusion.ddpm_sampling(x1=noisy_act, ema=self.ema,
                                                net=self.latent_policy,
                                                cond=past_cond.flatten(start_dim=1),
                                                diffuse_step=self.opt.interval)

        return pred_act

    def get_params(self):
        return self.parameters()


class DF_Policy_Agent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            totalset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            obs_mask_dim,
            eval_every_n_epochs: int = 50,
            normalize_input = True,
            test=False,
    ):
        super().__init__(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # Define the number of GPUs available
        # num_gpus = torch.cuda.device_count()

        # Check if multiple GPUs are available and select the appropriate device
        # if num_gpus > 1:
        #     print(f"Using {num_gpus} GPUs for training.")
        #     self.model = nn.DataParallel(self.model)

        # self.optimizer = hydra.utils.instantiate(optimization, params=self.model.parameters())


        self.eval_model_name = "eval_best_idm.pth"
        self.last_model_name = "last_idm.pth"

        print('init diffusion policy agent ')
        if not test:
            self.optimizer, self.sched = build_optimizer_sched(optimization, self.model.parameters(),self.train_dataloader, epoch)
        else:
            print("test initialize, skip optimizer")


        self.eval_model_name = "eval_best_idm.pth"
        self.last_model_name = "last_idm.pth"

        self.normalize_input = normalize_input
        self.obs_mask_dim = list(obs_mask_dim)

        self.totalset = hydra.utils.instantiate(totalset)
        self.Normalizer = Normalizer(self.totalset.get_all_observations(), self.totalset.get_all_actions(),
                                     self.normalize_input, device)


    def train_agent(self):
        best_test_mse = 1e10
        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch + 1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:

                    state, action, mask, _ = data
                    mean_mse = self.evaluate(state,action)

                    test_mse.append(mean_mse)

                avrg_test_mse = sum(test_mse) / len(test_mse)

                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    early_stop_cnt = 0
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')


                wandb.log(
                    {
                        "mean_test_loss": avrg_test_mse,
                    }
                )

            train_loss = []
            for data in self.train_dataloader:
                state, action, mask, _ = data
                batch_loss = self.train_step(state,action)

                train_loss.append(batch_loss)

                wandb.log(
                    {
                        "loss": batch_loss,
                    }
                )

            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}, lr_rate:{}".format(num_epoch, avrg_train_loss,
                                                                             self.sched.get_last_lr()[0]))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_vision_agent(self):

        pass

    def mask_act_from_state(self, state):
        if len(self.obs_mask_dim)==1:
            return state[:, :, self.obs_mask_dim[0]:]  # robot past action is removed
        elif len(self.obs_mask_dim) == 2:
            start, end = self.obs_mask_dim
            return torch.cat((state[:, :, :start], state[:, :, end:]), dim=2)  # mask out the specified part

        elif len(self.obs_mask_dim) > 2:
            mask = torch.ones(state.size(-1), dtype=torch.bool)
            for i in range(0, len(self.obs_mask_dim), 2):
                start = self.obs_mask_dim[i]
                end = self.obs_mask_dim[i + 1]
                mask[start:end] = False

            return state[:, :, mask]

        else:
            raise ValueError("obs_mask_dim should be either an integer or a list of two integers.")

    def train_step(self, state, action: Optional[torch.Tensor] = None, goal: Optional[torch.Tensor] = None):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()
        state = self.Normalizer.normalize_input(state)
        action = self.Normalizer.normalize_output(action)
        mstate = self.mask_act_from_state(state)
        loss = self.model.forward(mstate,action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.ema.update()

        self.sched.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, state, action: Optional[torch.Tensor] = None, goal: Optional[torch.Tensor] = None):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()
        self.model.ema.store()  # Store the current model parameters
        self.model.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:

            state = self.Normalizer.normalize_input(state)
            action = self.Normalizer.normalize_output(action)
            mstate = self.mask_act_from_state(state)

            obs_cond = mstate[:, 0:OBS_HORIZON]
            act_future = action[:, OBS_HORIZON:OBS_HORIZON + 12]

            pred_act_future = self.model.sample(obs_cond,act_future)

            mse = F.mse_loss(pred_act_future, act_future)

        finally:
            self.model.ema.restore()

        return mse.item()

    @torch.no_grad()
    def predict(self, state, if_vision=False,k_step=12) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()
        self.model.ema.store()  # Store the current model parameters
        self.model.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:
            state = self.Normalizer.normalize_input(state)
            mstate = self.mask_act_from_state(state)

            obs_cond = mstate[:, 0:OBS_HORIZON]

            dummy_act_future = torch.zeros((1,12,self.model.act_dim)).to(self.device)

            n_pred_act_future = self.model.sample(obs_cond, dummy_act_future)
            pred_actions = self.Normalizer.inverse_normalize_output(n_pred_act_future)

        finally:
            self.model.ema.restore()

        return pred_actions.cpu().numpy()[0]

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name
        ckpt_path = os.path.join(weights_path, file_name)
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.model.latent_policy.load_state_dict(ckpt["net"])
            self.model.ema.load_state_dict(ckpt["ema"])

            log.info(f'Loaded pre-trained idm policy from {ckpt_path}')
            log.info('loaded norm param from ckpt')

        except FileNotFoundError:
            log.error(f"Checkpoint file not found: {ckpt_path}")
            raise Exception(f"Checkpoint file not found: {ckpt_path}")

        except KeyError as e:
            log.error(f"Key error in checkpoint file: {e}")
            raise Exception(f"Key error in checkpoint file: {e}")

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name

        state_dict = {
            "net": self.model.latent_policy.state_dict(),
            "ema": self.model.ema.state_dict(),
        }

        torch.save(state_dict, os.path.join(store_path, file_name))

    def reset(self):
        pass