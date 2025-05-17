import os
import logging
import random

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="configs", config_name="aligning_config.yaml")
def main(cfg: DictConfig) -> None:

    # if cfg.seed in [10,20]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # elif cfg.seed in [30, 40]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # elif cfg.seed in [50,60]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    set_seed_everywhere(cfg.seed)

    ## init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        # mode="disabled",
        config=wandb.config,
        name=cfg.agent_name,
    )

    agent = hydra.utils.instantiate(cfg.agents)
    # train the agent
    agent.train_agent()

    # # load the model performs best on the evaluation set
    # agent.load_pretrained_model(agent.working_dir, sv_name=agent.eval_model_name)
    #
    # # simulate the model
    # env_sim = hydra.utils.instantiate(cfg.simulation)
    # env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()