import os
import logging
import pathlib

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from agents.utils.sim_path import sim_framework_path


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="aligning_config.yaml")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="disabled",
        config=wandb.config
    )

    agent = hydra.utils.instantiate(cfg.agents)

    cwd = pathlib.Path(__file__).parent.resolve()
    root = f'{cwd}/{cfg.log_dir}runs/'

    latent_policy_ckpt_path = root+ 'idm_latent_policy/'
    action_decoder_ckpt_path = root+ f'{cfg.agent_name}/'

    agent.load_model_from_ckpt(latent_policy_ckpt_path, action_decoder_ckpt_path)

    env_sim = hydra.utils.instantiate(cfg.simulation)
    success_rate = env_sim.test_agent(agent)
    print(f'eval success rate:{success_rate}')

    log.info("done")

    wandb.finish()

    return success_rate


if __name__ == "__main__":
    main()