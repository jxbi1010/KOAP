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
from agents.models.idm.util import WarmupLinearSchedule
from agents.models.idm.ema import ExponentialMovingAverage
log = logging.getLogger(__name__)

OBS_HORIZON = 2

class IDM_Baseline(nn.Module):

    def __init__(self,
                 model: DictConfig,
                 device: str = 'cuda'):

        super(IDM_Baseline, self).__init__()

        self.model = hydra.utils.instantiate(model).to(device)

    def forward(self, obs):

        pred = self.model.forward(obs)

        return pred

    def get_params(self):
        return self.parameters()


class IDM_Baseline_Agent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            totalset:DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            obs_mask_dim,
            eval_every_n_epochs: int = 50,
            normalize_input = True,
            scale_set = False,
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

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )
        self.scheduler = WarmupLinearSchedule(self.optimizer, 200, 0.1 * optimization.lr, optimization.lr)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.8)

        self.eval_model_name = "eval_best_idm.pth"
        self.last_model_name = "last_idm.pth"

        self.normalize_input = normalize_input
        self.obs_mask_dim = list(obs_mask_dim)

        self.totalset = hydra.utils.instantiate(totalset)
        self.Normalizer = Normalizer(self.totalset.get_all_observations(), self.totalset.get_all_actions(),
                                     self.normalize_input, device)


    def train_agent(self):
        best_test_mse = 1e10
        early_stop_cnt = 0
        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch+1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:
                    state, action, mask,_ = data #[torch.squeeze(data[i]) for i in range(3)]

                    mean_mse = self.evaluate(state, action)
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

                    # log.info('New best test loss. Stored weights have been updated!')
                else:
                    early_stop_cnt +=1
                wandb.log(
                    {
                        "mean_test_loss": avrg_test_mse,
                    }
                )
                if early_stop_cnt>=7:
                    log.info('Early Stop!')
                    break


            train_loss = []
            for data in self.train_dataloader:
                state, action, mask,_ = data

                batch_loss = self.train_step(state, action)

                train_loss.append(batch_loss)

                wandb.log({"loss": batch_loss,})

            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        print(f'save model to {self.working_dir}')
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
            # print(state.shape[-1])
        else:
            raise ValueError("obs_mask_dim should be either an integer or a list of two integers.")

    def train_step(self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        state = self.Normalizer.normalize_input(state)
        actions = self.Normalizer.normalize_output(actions)
        state = self.mask_act_from_state(state)

        pred_actions = self.model.forward(state)

        loss = F.mse_loss(pred_actions, actions[:,1:-1])

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.ema.update()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, state, action: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()
        self.ema.store()  # Store the current model parameters
        self.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:
            state = self.Normalizer.normalize_input(state)
            actions = self.Normalizer.normalize_output(action)
            state = self.mask_act_from_state(state)

            pred_actions = self.model.forward(state)

            mse = F.mse_loss(pred_actions, actions[:,1:-1])

        finally:
            self.ema.restore()

        return mse.item()

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()
        self.ema.store()  # Store the current model parameters
        self.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:
            state = self.Normalizer.normalize_input(state)
            state = self.mask_act_from_state(state)
            pred_actions = self.model.forward(state)
            pred_actions = self.Normalizer.inverse_normalize_output(pred_actions)

        finally:
            self.ema.restore()

        return pred_actions.detach().cpu().numpy()[0]

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name
        ckpt_path = os.path.join(weights_path, file_name)
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.model.model.load_state_dict(ckpt["net"])
            self.ema.load_state_dict(ckpt['ema'])

            log.info(f'Loaded pre-trained idm dynamics model from {ckpt_path}')

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
            "net": self.model.model.state_dict(),
            "ema": self.ema.state_dict(),
        }

        torch.save(state_dict, os.path.join(store_path, file_name))
            
    def reset(self):
        pass

