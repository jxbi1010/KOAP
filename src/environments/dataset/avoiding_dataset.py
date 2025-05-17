import random
from typing import Optional, Callable, Any
import logging

import os
import glob
import torch
import pickle
import numpy as np
import cv2

from environments.dataset.base_dataset import TrajectoryDataset
from agents.utils.sim_path import sim_framework_path


class Avoiding_Dataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 20,
            action_dim: int = 2,
            max_len_data: int = 256,
            window_size: int = 1,
            action_data_ratio: float = 100.0,  # percentage of action to mask
            obs_data_ratio: float = 100.0,
            only_label_data: bool = False,
            seed=None,
    ):

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size
        )

        logging.info("Loading Sorting Dataset")

        inputs = []
        actions = []
        masks = []

        data_dir = sim_framework_path(data_directory)
        state_files = os.listdir(data_dir)

        for file in state_files:
            with open(os.path.join(data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # robot and box posistion
            robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
            robot_c_pos = env_state['robot']['c_pos'][:, :2]

            input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)

            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
            valid_len = len(vel_state)

            zero_obs[0, :valid_len, :] = input_state[:-1]
            zero_action[0, :valid_len, :] = vel_state
            zero_mask[0, :valid_len] = 1

            inputs.append(zero_obs)
            actions.append(zero_action)
            masks.append(zero_mask)

        # shape: B, T, n
        self.observations = torch.from_numpy(np.concatenate(inputs)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.observations)

        self.action_data_ratio = action_data_ratio
        # self.slices = self.get_slices()

        # randomness control: make sure the generated mask is the same for all methods given the same seed.
        np.random.seed(seed)
        if action_data_ratio ==100:
            self.all_slices = self.get_slices()
            self.slices = [(s, True) for s in self.all_slices]
            np.random.shuffle(self.slices)  # Optionally shuffle here or ensure mixing during batching
            # print(f"initialize avoiding_dataset with all action data.")

        elif action_data_ratio ==0:
            self.all_slices = self.get_slices()
            self.slices = [(s, False) for s in self.all_slices]
            np.random.shuffle(self.slices)  # Optionally shuffle here or ensure mixing during batching
            # print(f"initialize avoiding_dataset with all action data masked.")

        else:
            self.active_slices, self.obs_slices = self.generate_slices(action_data_ratio, obs_data_ratio)

            if only_label_data:
                # use for generate eval set, set a different seed so that action access is different from training set.
                self.slices = [(s, True) for s in self.active_slices]

            else:
                self.slices = [(s, True) for s in self.active_slices] + [(s, False) for s in self.obs_slices]
                np.random.shuffle(self.slices)

                # first_ten_mask = np.array(active_slices)[0:10]
                # print(f"seed: {seed}, mask_id:{np.sum(first_ten_mask)}, num of slices:{len(self.slices)}")


    def generate_slices(self, action_data_ratio,obs_data_ratio):
        # all_slices = self.get_slices()  # Generate all possible slices first
        # num_active_slices = int(len(all_slices) * (action_data_ratio / 100))
        # active_indices = np.random.choice(len(all_slices), num_active_slices, replace=False)
        # active_slices = [all_slices[i] for i in active_indices]
        # # Create a set of all indices to easily filter out active ones
        # all_indices = set(range(len(all_slices)))
        # masked_indices = all_indices - set(active_indices)
        # masked_slices = [all_slices[i] for i in masked_indices]

        assert action_data_ratio <= obs_data_ratio

        all_slices = self.get_slices()  # Generate all possible slices first

        num_obs_slices = int(len(all_slices) * (obs_data_ratio / 100))
        num_active_slices = int(len(all_slices) * (action_data_ratio / 100))

        obs_indices = np.random.choice(len(all_slices), num_obs_slices, replace=False)
        act_indices = np.random.choice(obs_indices, num_active_slices, replace=False)
        obs_only_indices = np.setdiff1d(obs_indices, act_indices)

        obs_slices = [all_slices[i] for i in obs_only_indices]
        act_slices = [all_slices[i] for i in act_indices]

        print(f"use {action_data_ratio}%action data and {obs_data_ratio}%obs data for training.")
        print(
            f"create dataset: dataset total slices:{len(all_slices)}, total num of obs slices: {len(obs_slices) + len(act_slices)}, active num of slices: {len(act_slices)}, obs only slices:{len(obs_slices)}")
        return act_slices, obs_slices

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        try:
            # i, start, end = self.slices[idx]
            #
            # obs = self.observations[i, start:end]
            # act = self.actions[i, start:end]
            # mask = self.masks[i, start:end]
            #
            # return obs, act, mask
            slice_info, is_active = self.slices[idx]
            i, start, end = slice_info

            obs = self.observations[i, start:end]
            act = self.actions[i, start:end]
            mask = self.masks[i, start:end]

            return obs, act, mask, is_active

        except RuntimeError as e:
            logging.error(f"Error loading index {idx}: {e}")
            # Skip to the next index. If at the end, wrap around or return None.
            next_idx = idx + 1 if idx + 1 < len(self) else 0
            # Optionally, add a check to prevent infinite recursion.
            if next_idx == 0 or next_idx == idx:  # Simple safeguard, can be improved.
                raise ValueError("Unable to find a valid batch after a complete dataset pass.")
            return self.__getitem__(next_idx)  # Recursive call