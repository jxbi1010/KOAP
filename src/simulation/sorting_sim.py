import logging
import os

import multiprocessing as mp
import random

from envs.gym_sorting_env.gym_sorting.envs.sorting import Sorting_Env
import numpy as np
import torch
import wandb

from simulation.base_sim import BaseSim
from agents.utils.sim_path import sim_framework_path
import collections
log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class Sorting_Sim(BaseSim):
    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
            num_box: int = 2,
            if_vision: bool = False,
            max_steps_per_episode: int = 500
    ):
        super().__init__(seed, device, render, n_cores, if_vision)

        self.n_contexts = n_contexts
        self.n_trajectories_per_context = n_trajectories_per_context

        self.max_steps_per_episode = max_steps_per_episode

        self.num_box = num_box

        self.test_contexts = np.load(sim_framework_path("environments/dataset/data/sorting/" + str(num_box) + "_test_contexts.pkl"),
                                     allow_pickle=True)

        self.modes = np.load(sim_framework_path("environments/dataset/data/sorting/" + str(num_box) + "_mode_prob.pkl"), allow_pickle=True)

        self.mode_keys = np.array(list(self.modes.keys()))
        self.n_mode = len(self.mode_keys)

        mode_encoding = []

        for i in range(self.n_mode):
            mode_encoding.append(self.modes[self.mode_keys[i]])

        self.mode_encoding = torch.tensor(mode_encoding)

    def eval_agent(self, agent, contexts, n_trajectories, mode_encoding, successes, mean_distance, pid, cpu_set):

        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        env = Sorting_Env(max_steps_per_episode=self.max_steps_per_episode, render=self.render, num_boxes=self.num_box, if_vision=self.if_vision)
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        for context in contexts:
            for i in range(n_trajectories):

                agent.reset()

                # print(f'Context {context} Rollout {i}')
                # training contexts
                # env.manager.set_index(context)
                obs = env.reset(random=False, context=self.test_contexts[context], if_vision=self.if_vision)

                # obs = env.reset()

                # test contexts
                # test_context = env.manager.sample()
                # obs = env.reset(random=False, context=test_context)

                # rollout for image_based policy
                if self.if_vision:

                    robot_pos, bp_image, inhand_image = obs
                    bp_image = bp_image.transpose((2, 0, 1)) / 255.
                    inhand_image = inhand_image.transpose((2, 0, 1)) / 255.

                    fixed_z = env.robot_state()[2:]
                    des_robot_pos = robot_pos
                    done = False

                    while not done:

                        pred_action = agent.predict((bp_image, inhand_image, des_robot_pos), if_vision=self.if_vision)
                        pred_action = pred_action[0] + des_robot_pos

                        pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)
                        obs, reward, done, info = env.step(pred_action)

                        des_robot_pos = pred_action[:2]

                        robot_pos, bp_image, inhand_image = obs

                        # cv2.imshow('0', bp_image)
                        # cv2.waitKey(1)
                        #
                        # cv2.imshow('1', inhand_image)
                        # cv2.waitKey(1)

                        bp_image = bp_image.transpose((2, 0, 1)) / 255.
                        inhand_image = inhand_image.transpose((2, 0, 1)) / 255.

                else:

                    pred_action = env.robot_state()
                    fixed_z = pred_action[2:]
                    done = False
                    # while not done:
                    #     obs = np.concatenate((pred_action[:2], obs))
                    #
                    #     pred_action = agent.predict(obs)
                    #     pred_action = pred_action[0] + obs[:2]
                    #
                    #     pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)
                    #
                    #     obs, reward, done, info = env.step(pred_action)

                    obs = np.concatenate((pred_action[:2], obs))
                    obs_deque = collections.deque([obs] * 2, maxlen=2)

                    while not done:
                        obs_seq = np.stack(obs_deque)
                        pred_action_seq = agent.predict(obs_seq)

                        # for k in range(len(pred_action_seq)):
                        for k in range(4):
                            pred_action = pred_action_seq[k] + obs[:2]
                            pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)

                            obs, reward, done, info = env.step(pred_action)

                            obs = np.concatenate((pred_action[:2], obs))
                            obs_deque.append(obs)

                            if done:
                                break


                mode_encoding[context, i] = torch.tensor(info['mode'])
                successes[context, i] = torch.tensor(info['success'])
                # mean_distance[context, i] = torch.tensor(info['mean_distance'])

                suc = info['success']
                print(f'Context {context} Rollout {i} Success {suc}')

    ################################
    # we use multi-process for the simulation
    # n_contexts: the number of different contexts of environment
    # n_trajectories_per_context: test each context for n times, this is mostly used for multi-modal data
    # n_cores: the number of cores used for simulation
    ###############################
    def test_agent(self, agent):

        log.info('Starting trained model evaluation')

        mode_encoding = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        successes = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        mean_distance = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()

        contexts = np.arange(self.n_contexts)

        workload = self.n_contexts // self.n_cores

        num_cpu = mp.cpu_count()
        cpu_set = list(range(num_cpu))

        # start = self.seed * 20
        # end = start + 20
        #
        # cpu_set = cpu_set[start:end]
        print("there are cpus: ", num_cpu)
        ctx = mp.get_context('spawn')

        p_list = []
        if self.n_cores > 1:
            for i in range(self.n_cores):
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs={
                        "agent": agent,
                        "contexts": contexts[i * workload:(i + 1) * workload],
                        "n_trajectories": self.n_trajectories_per_context,
                        "mode_encoding": mode_encoding,
                        "successes": successes,
                        "mean_distance": mean_distance,
                        "pid": i,
                        "cpu_set": set(cpu_set[i:i+1])
                    },
                )
                print("Start {}".format(i))
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, contexts, self.n_trajectories_per_context, mode_encoding, successes, mean_distance, 0, cpu_set=set([0]))

        success_rate = torch.mean(successes).item()
        mode_probs = torch.zeros([self.n_contexts, self.n_mode])

        for c in range(self.n_contexts):

            for num in range(self.n_mode):
                mode_probs[c, num] = torch.tensor(
                    [sum(mode_encoding[c, successes[c, :] == 1] == self.mode_keys[num]) / self.n_trajectories_per_context])

        mode_probs /= (mode_probs.sum(1).reshape(-1, 1) + 1e-12)
        print(f'p(m|c) {mode_probs}')

        mode_probs = mode_probs[torch.nonzero(mode_probs.sum(1), as_tuple=True)[0]]

        entropy = - (mode_probs * torch.log(mode_probs + 1e-12) / torch.log(torch.tensor(self.n_mode))).sum(1).mean()
        log_ = (mode_probs * torch.log(self.mode_encoding + 1e-12) / torch.log(torch.tensor(self.n_mode))).sum(1).mean()

        KL = - entropy - log_

        wandb.log({'score': (success_rate - KL)})
        wandb.log({'Metrics/successes': success_rate})
        wandb.log({'Metrics/KL': KL})
        wandb.log({'Metrics/entropy': entropy})
        # wandb.log({'Metrics/distance': mean_distance.mean().item()})

        # print(f'Mean Distance {mean_distance.mean().item()}')
        print(f'Successrate {success_rate}')
        print(f'entropy {entropy}')
        print(f'KL {KL}')

        log.info(f'Successrate {success_rate}')
        log.info(f'entropy {entropy}')
        log.info(f'KL {KL}')

        return success_rate