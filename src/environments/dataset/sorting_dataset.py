import random
from typing import Optional, Callable, Any
import logging

import os
import glob
import torch
import pickle
import numpy as np
import cv2
from tqdm import tqdm

from environments.dataset.base_dataset import TrajectoryDataset
from agents.utils.sim_path import sim_framework_path
from .geo_transform import quat2euler


class Sorting_Dataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 20,
            action_dim: int = 2,
            max_len_data: int = 256,
            window_size: int = 1,
            num_boxes: int = 2,
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

        # for root, dirs, files in os.walk(self.data_directory):
        #
        #     for mode_dir in dirs:

        # state_files = glob.glob(os.path.join(root, mode_dir) + "/env*")
        # data_dir = os.path.join(sim_framework_path(data_directory), "local")
        # data_dir = sim_framework_path(data_directory)
        # state_files = glob.glob(data_dir + "/env*")

        # random.seed(0)

        # data_dir = sim_framework_path(data_directory)
        # state_files = os.listdir(data_dir)
        #
        # random.shuffle(state_files)
        #
        # if data == "train":
        #     env_state_files = state_files[50:]
        # elif data == "eval":
        #     env_state_files = state_files[:50]
        # else:
        #     assert False, "wrong data type"

        data_dir = sim_framework_path(data_directory)
        print(f"data_dir:{data_dir}")
        if os.path.isdir(data_dir):
            state_files = os.listdir(data_dir)

        elif os.path.isfile(data_dir):
            state_files = np.load(data_dir, allow_pickle=True)

            if num_boxes == 2:
                data_dir = sim_framework_path("environments/dataset/data/sorting/2_boxes/state")
            elif num_boxes == 4:
                data_dir = sim_framework_path("environments/dataset/data/sorting/4_boxes/state")
            elif num_boxes == 6:
                data_dir = sim_framework_path("environments/dataset/data/sorting/6_boxes/state")
            else:
                assert False, "check num boxes"

            print(f"data_dir2:{data_dir}")
        #
        # state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        for file in state_files:
            with open(os.path.join(data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            # lengths.append(len(env_state['robot']['des_c_pos']))

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # robot and box posistion
            robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
            robot_c_pos = env_state['robot']['c_pos'][:, :2]

            if num_boxes == 2:
                red_box1_pos = env_state['red-box1']['pos'][:, :2]
                red_box1_quat = np.tan(quat2euler(env_state['red-box1']['quat'])[:, -1:])

                blue_box1_pos = env_state['blue-box1']['pos'][:, :2]
                blue_box1_quat = np.tan(quat2euler(env_state['blue-box1']['quat'])[:, -1:])

                input_state = np.concatenate((robot_des_pos, robot_c_pos, red_box1_pos, red_box1_quat,
                                              blue_box1_pos, blue_box1_quat), axis=-1)

            elif num_boxes == 4:

                red_box1_pos = env_state['red-box1']['pos'][:, :2]
                red_box1_quat = np.tan(quat2euler(env_state['red-box1']['quat'])[:, -1:])

                red_box2_pos = env_state['red-box2']['pos'][:, :2]
                red_box2_quat = np.tan(quat2euler(env_state['red-box2']['quat'])[:, -1:])

                blue_box1_pos = env_state['blue-box1']['pos'][:, :2]
                blue_box1_quat = np.tan(quat2euler(env_state['blue-box1']['quat'])[:, -1:])

                blue_box2_pos = env_state['blue-box2']['pos'][:, :2]
                blue_box2_quat = np.tan(quat2euler(env_state['blue-box2']['quat'])[:, -1:])

                input_state = np.concatenate((robot_des_pos, robot_c_pos, red_box1_pos, red_box1_quat,
                                              red_box2_pos, red_box2_quat, blue_box1_pos, blue_box1_quat,
                                              blue_box2_pos, blue_box2_quat), axis=-1)

            elif num_boxes == 6:

                red_box1_pos = env_state['red-box1']['pos'][:, :2]
                red_box1_quat = np.tan(quat2euler(env_state['red-box1']['quat'])[:, -1:])

                red_box2_pos = env_state['red-box2']['pos'][:, :2]
                red_box2_quat = np.tan(quat2euler(env_state['red-box2']['quat'])[:, -1:])

                red_box3_pos = env_state['red-box3']['pos'][:, :2]
                red_box3_quat = np.tan(quat2euler(env_state['red-box3']['quat'])[:, -1:])

                blue_box1_pos = env_state['blue-box1']['pos'][:, :2]
                blue_box1_quat = np.tan(quat2euler(env_state['blue-box1']['quat'])[:, -1:])

                blue_box2_pos = env_state['blue-box2']['pos'][:, :2]
                blue_box2_quat = np.tan(quat2euler(env_state['blue-box2']['quat'])[:, -1:])

                blue_box3_pos = env_state['blue-box3']['pos'][:, :2]
                blue_box3_quat = np.tan(quat2euler(env_state['blue-box3']['quat'])[:, -1:])

                input_state = np.concatenate((robot_des_pos, robot_c_pos, red_box1_pos, red_box1_quat, red_box2_pos, red_box2_quat,
                                              red_box3_pos, red_box3_quat, blue_box1_pos, blue_box1_quat, blue_box2_pos, blue_box2_quat,
                                              blue_box3_pos, blue_box3_quat), axis=-1)

            else:
                assert False, "check num boxes"

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

        # randomness control: make sure the generated mask is the same for all methods given the same seed.
        np.random.seed(seed)

        self.action_data_ratio = action_data_ratio
        # self.slices = self.get_slices()
        if action_data_ratio == 100:
            self.all_slices = self.get_slices()
            self.slices = [(s, True) for s in self.all_slices]
            np.random.shuffle(self.slices)  # Optionally shuffle here or ensure mixing during batching
            # print(f"initialize aligning_dataset with all action data.")

        elif action_data_ratio == 0:
            self.all_slices = self.get_slices()
            self.slices = [(s, False) for s in self.all_slices]
            np.random.shuffle(self.slices)  # Optionally shuffle here or ensure mixing during batching
            # print(f"initialize aligning_dataset with all action data masked.")

        else:
            self.active_slices, self.obs_slices = self.generate_slices(action_data_ratio, obs_data_ratio)

            if only_label_data:
                self.slices = [(s, True) for s in self.active_slices]
            else:
                self.slices = [(s, True) for s in self.active_slices] + [(s, False) for s in self.obs_slices]
                np.random.shuffle(self.slices)  # Optionally shuffle here or ensure mixing during batching

            # print(np.array(active_slices))
            # np.save('mask_file.npy', np.array(active_slices))
            # print('save mask file')
            print(f"total number of sequences:{len(self.slices)}")


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
        print(f"create dataset: dataset total slices:{len(all_slices)}, total num of obs slices: {len(obs_slices)+len(act_slices)}, active num of slices: {len(act_slices)}, obs only slices:{len(obs_slices)}")
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

            slice_info, is_active = self.slices[idx]
            i, start, end = slice_info

            obs = self.observations[i, start:end]
            act = self.actions[i, start:end]
            mask = self.masks[i, start:end]

            # Calculate the actual length of the sequence
            seq_len = end - start

            # Check if the sequence is shorter than the window size
            if seq_len < self.window_size:
                print("[Data_Loader: seq_length < window_size, padding  ]")
                # Calculate how much padding is needed
                padding_len = self.window_size - seq_len

                # Replicate the last timestep in each sequence for padding
                obs_pad = obs[:, -1, :].unsqueeze(1).repeat(1, padding_len, 1)
                act_pad = act[:, -1, :].unsqueeze(1).repeat(1, padding_len, 1)
                mask_pad = mask[:, -1].unsqueeze(1).repeat(1, padding_len)

                obs = torch.cat([obs, obs_pad], dim=1)
                act = torch.cat([act, act_pad], dim=1)
                mask = torch.cat([mask, mask_pad], dim=1)

            return obs, act, mask, is_active

        except RuntimeError as e:
            logging.error(f"Error loading index {idx}: {e}")
            # Skip to the next index. If at the end, wrap around or return None.
            next_idx = idx + 1 if idx + 1 < len(self) else 0
            # Optionally, add a check to prevent infinite recursion.
            if next_idx == 0 or next_idx == idx:  # Simple safeguard, can be improved.
                raise ValueError("Unable to find a valid batch after a complete dataset pass.")
            return self.__getitem__(next_idx)  # Recursive call


class Sorting_Img_Dataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 20,
            action_dim: int = 2,
            max_len_data: int = 256,
            window_size: int = 1,
            num_boxes: int = 2
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

        # for root, dirs, files in os.walk(self.data_directory):
        #
        #     for mode_dir in dirs:

        # state_files = glob.glob(os.path.join(root, mode_dir) + "/env*")
        # data_dir = os.path.join(sim_framework_path(data_directory), "local")
        # data_dir = sim_framework_path(data_directory)
        # state_files = glob.glob(data_dir + "/env*")

        # random.seed(0)
        #
        # data_dir = sim_framework_path(data_directory)
        # state_files = glob.glob(data_dir + '/state/*')
        #
        # random.shuffle(state_files)
        #
        # if data == "train":
        #     env_state_files = state_files[30:]
        # elif data == "eval":
        #     env_state_files = state_files[:30]
        # else:
        #     assert False, "wrong data type"

        if num_boxes == 2:
            data_dir = sim_framework_path("environments/dataset/data/sorting/2_boxes/")
        elif num_boxes == 4:
            data_dir = sim_framework_path("environments/dataset/data/sorting/4_boxes/")
        elif num_boxes == 6:
            data_dir = sim_framework_path("environments/dataset/data/sorting/6_boxes/")
        else:
            assert False, "check num boxes"

        state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        bp_cam_imgs = []
        inhand_cam_imgs = []

        for file in tqdm(state_files[:100]):
            with open(os.path.join(data_dir, 'state', file), 'rb') as f:
                env_state = pickle.load(f)

            # lengths.append(len(env_state['robot']['des_c_pos']))
            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # robot and box posistion
            robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
            robot_c_pos = env_state['robot']['c_pos'][:, :2]

            file_name = os.path.basename(file).split('.')[0]

            ###############################################################
            bp_images = []
            bp_imgs = glob.glob(data_dir + '/images/bp-cam/' + file_name + '/*')
            bp_imgs.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

            for img in bp_imgs:
                image = cv2.imread(img).astype(np.float32)
                image = image.transpose((2, 0, 1)) / 255.

                image = torch.from_numpy(image).to(self.device).float().unsqueeze(0)

                bp_images.append(image)

            bp_images = torch.concatenate(bp_images, dim=0)
            ################################################################
            inhand_imgs = glob.glob(data_dir + '/images/inhand-cam/' + file_name + '/*')
            inhand_imgs.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            inhand_images = []
            for img in inhand_imgs:
                image = cv2.imread(img).astype(np.float32)
                image = image.transpose((2, 0, 1)) / 255.

                image = torch.from_numpy(image).to(self.device).float().unsqueeze(0)

                inhand_images.append(image)
            inhand_images = torch.concatenate(inhand_images, dim=0)
            ##################################################################
            # input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)

            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]

            valid_len = len(vel_state)

            zero_obs[0, :valid_len, :] = robot_des_pos[:-1]
            zero_action[0, :valid_len, :] = vel_state
            zero_mask[0, :valid_len] = 1

            bp_cam_imgs.append(bp_images)
            inhand_cam_imgs.append(inhand_images)

            inputs.append(zero_obs)
            actions.append(zero_action)
            masks.append(zero_mask)

        self.bp_cam_imgs = bp_cam_imgs
        self.inhand_cam_imgs = inhand_cam_imgs

        # shape: B, T, n
        self.observations = torch.from_numpy(np.concatenate(inputs)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.actions)

        self.slices = self.get_slices()

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

        i, start, end = self.slices[idx]

        obs = self.observations[i, start:end]
        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        bp_imgs = self.bp_cam_imgs[i][start:end]
        inhand_imgs = self.inhand_cam_imgs[i][start:end]

        # bp_imgs = np.zeros((self.window_size, 3, 96, 96), dtype=np.float32)
        # inhand_imgs = np.zeros((self.window_size, 3, 96, 96), dtype=np.float32)
        #
        # for num_frame, img_file in enumerate(bp_img_files):
        #     image = cv2.imread(img_file).astype(np.float32)
        #     bp_imgs[num_frame] = image.transpose((2, 0, 1)) / 255.
        #
        # for num_frame, img_file in enumerate(inhand_img_files):
        #     image = cv2.imread(img_file).astype(np.float32)
        #     inhand_imgs[num_frame] = image.transpose((2, 0, 1)) / 255.
        #
        # bp_imgs = torch.from_numpy(bp_imgs).to(self.device).float()
        # inhand_imgs = torch.from_numpy(inhand_imgs).to(self.device).float()

        return bp_imgs, inhand_imgs, obs, act, mask