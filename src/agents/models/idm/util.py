# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
from torch.utils.tensorboard import SummaryWriter
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.optim import AdamW, lr_scheduler
import agents.models.idm.distributed_util as dist_util
import numpy as np
import random
from diffusers.optimization import get_scheduler

from torch.optim.lr_scheduler import LRScheduler
import math


def block_diag(*arrs):
    """Construct a block diagonal matrix from the given matrices."""
    shapes = [a.shape for a in arrs]
    out_shape = [sum([shape[0] for shape in shapes]), sum([shape[1] for shape in shapes])]

    out = torch.zeros(out_shape, dtype=arrs[0].dtype)

    r, c = 0, 0
    for a in arrs:
        rr, cc = a.shape
        out[r:r + rr, c:c + cc] = a
        r += rr
        c += cc

    return out


def block_diag_3d(batch_size, *arrs):
    """Construct a block diagonal matrix from the given matrices for batch processing."""
    shapes = [a.shape for a in arrs]
    out_shape = [batch_size, sum([shape[1] for shape in shapes]), sum([shape[2] for shape in shapes])]

    out = torch.zeros(out_shape, dtype=arrs[0].dtype, device=arrs[0].device)

    r, c = 0, 0
    for a in arrs:
        rr, cc = a.shape[1], a.shape[2]
        out[:, r:r + rr, c:c + cc] = a
        r += rr
        c += cc

    return out


class WarmupLinearSchedule(LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_lr, max_lr, decay_steps=100, decay_rate=0.995, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear increase from base_lr to max_lr during warmup
            lr_scale = self.base_lr + (self.max_lr - self.base_lr) * (float(self.last_epoch) / self.warmup_steps)
            return [lr_scale for _ in self.base_lrs]
        else:
            # Exponential decay after warmup
            # decay_factor = self.decay_rate ** ((self.last_epoch - self.warmup_steps) / self.decay_steps)
            # return [self.max_lr * decay_factor for _ in self.base_lrs]
            return [self.max_lr for _ in self.base_lrs]

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()


def build_optimizer_sched(opt, params,dataloader,total_epochs):
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.weight_decay}
    optimizer = AdamW(params, **optim_dict)

    warm_up_ratio = 0.05
    total_training_step = total_epochs*len(dataloader)
    warm_up_step = warm_up_ratio*total_training_step

    sched = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=warm_up_step,
        num_training_steps=total_training_step
    )

    return optimizer, sched



class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def setup_loader(dataset, batch_size, num_workers=4):
    loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

    while True:
        yield from loader

class BaseWriter(object):
    def __init__(self, opt):
        self.rank = opt.global_rank
    def add_scalar(self, step, key, val):
        pass # do nothing
    def add_image(self, step, key, image):
        pass # do nothing
    def close(self): pass

class WandBWriter(BaseWriter):
    def __init__(self, opt):
        super(WandBWriter,self).__init__(opt)
        if self.rank == 0:
            assert wandb.login(key=opt.wandb_api_key)
            wandb.init(dir=str(opt.log_dir), project="i2sb", entity=opt.wandb_user, name=opt.name, config=vars(opt))

    def add_scalar(self, step, key, val):
        if self.rank == 0: wandb.log({key: val}, step=step)

    def add_image(self, step, key, image):
        if self.rank == 0:
            # adopt from torchvision.utils.save_image
            image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            wandb.log({key: wandb.Image(image)}, step=step)


class TensorBoardWriter(BaseWriter):
    def __init__(self, opt):
        super(TensorBoardWriter,self).__init__(opt)
        if self.rank == 0:
            run_dir = str(opt.log_dir / opt.name)
            os.makedirs(run_dir, exist_ok=True)
            self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        if self.rank == 0: self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        if self.rank == 0:
            image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
            self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        if self.rank == 0: self.writer.close()

def build_log_writer(opt):
    if opt.log_writer == 'wandb': return WandBWriter(opt)
    elif opt.log_writer == 'tensorboard': return TensorBoardWriter(opt)
    else: return BaseWriter(opt) # do nothing

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def apply_kaiming_normal(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):  # Check if it is a convolutional or linear layer
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Optionally initialize biases to 0



if __name__ == "__main__":
    K1 = torch.randn(3, 3)
    K2 = torch.randn(2, 2)
    K3 = torch.randn(1, 1)

    # Construct the block diagonal matrix
    K_block_diag = block_diag(K1, K2, K3)
    print(K_block_diag)