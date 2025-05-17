import math
import random
from typing import Optional, Callable, Any
import logging

import os
import glob

import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset
from environments.dataset.base_dataset import TrajectoryDataset
from agents.utils.sim_path import sim_framework_path

# from .geo_transform import quat2euler
from environments.dataset.geo_transform import quat2euler

class Stacking_Dataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 20,
            action_dim: int = 2,
            max_len_data: int = 256,
            window_size: int = 1,
            action_data_ratio: float = 100.0,  # percentage of action to mask
            obs_data_ratio: float = 50.0,
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

        logging.info("Loading CubeStacking Dataset")

        inputs = []
        actions = []
        masks = []

        data_dir = sim_framework_path(data_directory)


        # for root, dirs, files in os.walk(self.data_directory):
        #
        #     for mode_dir in dirs:

        # state_files = glob.glob(os.path.join(root, mode_dir) + "/env*")
        # data_dir = os.path.join(sim_framework_path(data_directory), "local")
        # data_dir = sim_framework_path(data_directory)
        # state_files = glob.glob(data_dir + "/env*")

        # bp_data_dir = sim_framework_path("environments/dataset/data/stacking/all_data_new")
        # state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        # bp_data_dir = sim_framework_path("environments/dataset/data/stacking/single_test")
        # state_files = os.listdir(bp_data_dir)

        # random.seed(0)
        #
        # data_dir = sim_framework_path(data_directory)
        # state_files = os.listdir(data_dir)
        #
        # random.shuffle(state_files)
        #
        # if data == "train":
        #     env_state_files = state_files[20:]
        # elif data == "eval":
        #     env_state_files = state_files[:20]
        # else:
        #     assert False, "wrong data type"

        if os.path.isdir(data_dir):
            state_files = os.listdir(data_dir)

        elif os.path.isfile(data_dir):
            state_files = np.load(data_dir, allow_pickle=True)
            data_dir = sim_framework_path("environments/dataset/data/stacking/all_data/")

        # data_dir = sim_framework_path("environments/dataset/data/stacking/all_data")
        # state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        for file in state_files:
            with open(os.path.join(data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            # lengths.append(len(env_state['robot']['des_c_pos']))

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # robot and box positions
            robot_des_j_pos = env_state['robot']['des_j_pos']
            robot_des_j_vel = env_state['robot']['des_j_vel']

            robot_des_c_pos = env_state['robot']['des_c_pos']
            robot_des_quat = env_state['robot']['des_c_quat']

            robot_c_pos = env_state['robot']['c_pos']
            robot_c_quat = env_state['robot']['c_quat']

            robot_j_pos = env_state['robot']['j_pos']
            robot_j_vel = env_state['robot']['j_vel']

            robot_gripper = np.expand_dims(env_state['robot']['gripper_width'], -1)
            # pred_gripper = np.zeros(robot_gripper.shape, dtype=np.float32)
            # pred_gripper[robot_gripper > 0.075] = 1

            sim_steps = np.expand_dims(np.arange(len(robot_des_j_pos)), -1)

            red_box_pos = env_state['red-box']['pos']
            red_box_quat = np.tan(quat2euler(env_state['red-box']['quat'])[:, -1:])
            # red_box_quat = np.concatenate((np.sin(red_box_quat), np.cos(red_box_quat)), axis=-1)

            green_box_pos = env_state['green-box']['pos']
            green_box_quat = np.tan(quat2euler(env_state['green-box']['quat'])[:, -1:])
            # green_box_quat = np.concatenate((np.sin(green_box_quat), np.cos(green_box_quat)), axis=-1)

            blue_box_pos = env_state['blue-box']['pos']
            blue_box_quat = np.tan(quat2euler(env_state['blue-box']['quat'])[:, -1:])
            # blue_box_quat = np.concatenate((np.sin(blue_box_quat), np.cos(blue_box_quat)), axis=-1)

            # target_box_pos = env_state['target-box']['pos'] #- robot_c_pos

            # input_state = np.concatenate((robot_des_c_pos, robot_des_quat, pred_gripper, red_box_pos, red_box_quat), axis=-1)

            # input_state = np.concatenate((robot_des_j_pos, robot_gripper, blue_box_pos, blue_box_quat), axis=-1)

            input_state = np.concatenate((robot_des_j_pos, robot_gripper, red_box_pos, red_box_quat, green_box_pos, green_box_quat,
                                          blue_box_pos, blue_box_quat), axis=-1)

            # input_state = np.concatenate((robot_des_j_pos, robot_des_j_vel, robot_c_pos, robot_c_quat, green_box_pos, green_box_quat,
            #                               target_box_pos), axis=-1)

            vel_state = robot_des_j_pos[1:] - robot_des_j_pos[:-1]

            valid_len = len(input_state) - 1

            zero_obs[0, :valid_len, :] = input_state[:-1]
            zero_action[0, :valid_len, :] = np.concatenate((vel_state, robot_gripper[1:]), axis=-1)
            zero_mask[0, :valid_len] = 1

            inputs.append(zero_obs)
            actions.append(zero_action)
            masks.append(zero_mask)

        # shape: B, T, n
        self.observations = torch.from_numpy(np.concatenate(inputs)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.observations)

        # randomness control: make sure the generated mask is the same for all methods given the same seed.
        np.random.seed(seed)


        self.action_data_ratio = action_data_ratio
        # self.slices = self.get_slices()
        if action_data_ratio ==100:
            self.all_slices = self.get_slices()
            self.slices = [(s, True) for s in self.all_slices]
            np.random.shuffle(self.slices)  # Optionally shuffle here or ensure mixing during batching
            print(f"initialize stacking_dataset with all action data.")

        elif action_data_ratio ==0:
            self.all_slices = self.get_slices()
            self.slices = [(s, False) for s in self.all_slices]
            np.random.shuffle(self.slices)  # Optionally shuffle here or ensure mixing during batching
            print(f"initialize stacking_dataset with all action data masked.")

        else:
            self.active_slices, self.obs_slices = self.generate_slices(action_data_ratio, obs_data_ratio)

            if only_label_data:
                self.slices = [(s, True) for s in self.active_slices]
            else:
                self.slices = [(s, True) for s in self.active_slices] + [(s, False) for s in self.obs_slices]
                np.random.shuffle(self.slices)  # Optionally shuffle here or ensure mixing during batching

            # print(np.array(active_slices))
            # print(len(self.slices))

    def generate_slices(self, action_data_ratio, obs_data_ratio):
        assert action_data_ratio<=obs_data_ratio

        all_slices = self.get_slices()  # Generate all possible slices first

        num_obs_slices = int(len(all_slices) * (obs_data_ratio / 100))
        num_active_slices = int(len(all_slices) * (action_data_ratio / 100))

        obs_indices = np.random.choice(len(all_slices), num_obs_slices, replace=False)
        act_indices = np.random.choice(obs_indices, num_active_slices, replace=False)
        obs_only_indices = np.setdiff1d(obs_indices, act_indices)

        obs_slices = [all_slices[i] for i in obs_only_indices]
        act_slices = [all_slices[i] for i in act_indices]

        print(f"use {action_data_ratio}%action data and {obs_data_ratio}%obs data for training.")
        print(f"create dataset: dataset total slices:{len(all_slices)}, total num of obs slices: {len(obs_slices) + len(act_slices)}, active num of slices: {len(act_slices)}, obs only slices:{len(obs_slices)}")
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


if __name__=="__main__":
    stack_dataset = Stacking_Dataset(data_directory='environments/dataset/data/stacking/train_files.pkl',
                                     seed=10)



