import logging
import os
import pathlib
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
# from torch_ema import ExponentialMovingAverage
from agents.models.idm.ema import ExponentialMovingAverage
from agents.models.idm.util import WarmupLinearSchedule

log = logging.getLogger(__name__)

OBS_HORIZON = 2

class IDM_Cotrain_Policy(nn.Module):

    def __init__(self,
                 idm_model: DictConfig,
                 device: str = 'cuda:0'):

        super(IDM_Cotrain_Policy, self).__init__()

        self.idm_model = hydra.utils.instantiate(idm_model).to(device)

    def forward(self,nobs):

        x,act = self.idm_model.forward(nobs)

        return x,act

    def inference(self,nobs):

        act = self.idm_model.inference(nobs)

        return act

    def get_params(self):
        return self.parameters()



class IDM_Cotrain_Agent(BaseAgent):
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
            normalize_input=True,
            action_loss_factor:float = 0.01,
    ):
        super().__init__(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # # Define the number of GPUs available
        # num_gpus = torch.cuda.device_count()
        #
        # # Check if multiple GPUs are available and select the appropriate device
        # if num_gpus > 1:
        #     print(f"Using {num_gpus} GPUs for training.")
        #     self.model = nn.DataParallel(self.model)

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )

        self.scheduler = WarmupLinearSchedule(self.optimizer, 200, 0.1*optimization.lr, optimization.lr)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.8)

        self.eval_model_name = "eval_best_idm.pth"
        self.last_model_name = "last_idm.pth"

        # self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        # self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.normalize_input = normalize_input
        self.obs_mask_dim = obs_mask_dim

        self.totalset = hydra.utils.instantiate(totalset)
        self.Normalizer = Normalizer(self.totalset.get_all_observations(), self.totalset.get_all_actions(),
                                     self.normalize_input, device)

        self.idm_model_loaded= False

        self.action_loss_factor = action_loss_factor
        print(f'init idm_model contrain agent, with action_loss_factor:{self.action_loss_factor}')

    def train_agent(self):
        best_test_mse = 1e10
        early_stop_cnt = 0
        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch+1) % self.eval_every_n_epochs:
                test_mse = []
                test_obs_mse = []
                test_act_mse = []
                for data in self.test_dataloader:
                    state, action, mask, is_active = data #[torch.squeeze(data[i]) for i in range(3)]
                    is_active = torch.as_tensor(is_active, dtype=torch.bool).to(self.device)
                    mean_mse,obs_mse,act_mse = self.evaluate(state, action, is_active)

                    test_mse.append(mean_mse)
                    test_obs_mse.append(obs_mse)
                    test_act_mse.append(act_mse)

                avrg_test_mse = sum(test_mse) / len(test_mse)
                avrg_obs_mse = sum(test_obs_mse) / len(test_obs_mse)
                avrg_act_mse = sum(test_act_mse) / len(test_act_mse)

                log.info("Epoch {}: Mean test act mse is {}".format(num_epoch,avrg_act_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    early_stop_cnt=0
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch,
                        }
                    )

                    # log.info('New best test loss. Stored weights have been updated!')
                else:
                    early_stop_cnt +=1
                wandb.log(
                    {
                        "test_total_loss": avrg_test_mse,
                        "test_obs_loss": avrg_obs_mse,
                        "test_act_loss": avrg_act_mse,
                    }
                )
                if early_stop_cnt>=7:
                    log.info('Early Stop!')
                    break

            train_loss = []
            obs_loss = []
            act_loss = []
            for data in self.train_dataloader:
                state, action, mask, is_active = data #[torch.squeeze(data[i]) for i in range(3)]
                is_active = torch.as_tensor(is_active, dtype=torch.bool).to(self.device)
                batch_loss,batch_obs_loss,batch_act_loss = self.train_step(state, action, is_active)

                train_loss.append(batch_loss)
                obs_loss.append(batch_obs_loss)
                act_loss.append(batch_act_loss)

            avrg_train_loss = sum(train_loss) / len(train_loss)
            avrg_obs_loss = sum(obs_loss) / len(obs_loss)
            avrg_act_loss= sum(act_loss) / len(act_loss)

            wandb.log(
                {
                    "train_total_loss": avrg_train_loss,
                    "train_obs_loss": avrg_obs_loss,
                    "train_act_loss": avrg_act_loss,
                }
            )

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

    def train_step(self, state, actions: torch.Tensor, is_active: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        nstate = self.Normalizer.normalize_input(state)
        naction = self.Normalizer.normalize_output(actions)
        nstate = self.mask_act_from_state(nstate)

        pred_observations, pred_actions = self.model.forward(nstate)

        obs_recon_loss = F.mse_loss(pred_observations, nstate[:, OBS_HORIZON:])

        act_pred_loss = F.mse_loss(pred_actions, naction[:,1:-1], reduction='none').mean(dim=(1,2)) # mean over time and feature dimension
        act_pred_loss = act_pred_loss*is_active
        active_sum = is_active.sum() + 1e-8
        act_pred_loss = self.action_loss_factor*act_pred_loss.sum()/active_sum

        loss = obs_recon_loss + act_pred_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.ema.update()
        self.scheduler.step()

        return loss.item(),obs_recon_loss.item(),act_pred_loss.item()

    @torch.no_grad()
    def evaluate(self, state, action: torch.Tensor, is_active: torch.Tensor):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()
        self.ema.store()  # Store the current model parameters
        self.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:
            nstate = self.Normalizer.normalize_input(state)
            naction = self.Normalizer.normalize_output(action)
            nstate = self.mask_act_from_state(nstate)

            pred_observations, pred_actions = self.model.forward(nstate)

            obs_recon_loss = F.mse_loss(pred_observations, nstate[:, OBS_HORIZON:])

            act_pred_loss = self.action_loss_factor*F.mse_loss(pred_actions, naction[:,1:-1])

            loss = act_pred_loss

        finally:
            self.ema.restore()

        return loss.item(),obs_recon_loss.item(),act_pred_loss.item()

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()
        self.ema.store()  # Store the current model parameters
        self.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:
            nstate = self.Normalizer.normalize_input(state)
            nstate = self.mask_act_from_state(nstate)
            pred_action = self.model.inference(nstate)

            pred_action = self.Normalizer.inverse_normalize_output(pred_action)

        finally:
            self.ema.restore()
        # pred_action = pred_action.clamp_(self.min_action, self.max_action)

        return pred_action.detach().cpu().numpy()[0]

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name
        ckpt_path = os.path.join(weights_path, file_name)
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.model.idm_model.load_state_dict(ckpt["net"])
            self.ema.load_state_dict(ckpt['ema'])
            log.info(f'Loaded pre-trained idm_model from {ckpt_path}')



        except FileNotFoundError:
            log.error(f"Checkpoint file not found: {ckpt_path}")
            raise Exception(f"Checkpoint file not found: {ckpt_path}")
        except KeyError as e:
            log.error(f"Key error in checkpoint file: {e}")

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name

        state_dict = {
            "net": self.model.idm_model.state_dict(),
            "ema": self.ema.state_dict(),
        }

        torch.save(state_dict, os.path.join(store_path, file_name))


            
    def reset(self):
        pass