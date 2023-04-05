import os

import seaborn as sn
import torch
from IPython.core.display import display
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from model import DQNLightning

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

def main():
    model = DQNLightning()

    trainer = Trainer(
        accelerator="tpu",
        devices=1,  # limiting got iPython runs
        max_epochs=150,
        val_check_interval=50,
        logger=CSVLogger(save_dir="logs/"),
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()