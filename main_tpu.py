"""Deep Reinforcement Learning: Deep Q-network (DQN)

The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.

To run the template, just run:
`python main_tpu.py`

The total_reward should the not surpass a score of 500.

References
----------

[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
"""

import os, time, math
import argparse

import torch
from IPython.core.display import display
from lightning.pytorch import cli_lightning_logo, seed_everything, Trainer
from lightning.pytorch.loggers import CSVLogger

from model import DQNLightning

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

def main(args):

    trainer = Trainer(accelerator="tpu", val_check_interval=50, max_epochs=200, devices=args.devices)
    del args.devices
    model = DQNLightning(**vars(args))
    
    start = time.time()
    trainer.fit(model)
    end = time.time()

    print('Training complete, time: ' + str(round(end - start, 2)) + 's')
    

if __name__ == "__main__":
    cli_lightning_logo()
    seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment tag")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
    parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
    parser.add_argument(
        "--warm_start_steps",
        type=int,
        default=1000,
        help="how many samples do we use to fill our buffer at the start of training",
    )
    parser.add_argument("--eps_last_frame", type=int, default=1000, help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
    parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
    parser.add_argument("--devices", type=int, default=1, help="number of TPU devices")

    args = parser.parse_args()

    main(args)