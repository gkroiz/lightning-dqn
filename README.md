# Pytorch Lightning Deep Q Network (DQN) for TPUs
This repository is based on the reinforcement DQN [model](https://github.com/Lightning-AI/lightning/blob/master/examples/pytorch/domain_templates/reinforce_learn_Qnet.py). There is also a tutorial from the [lightning documentation](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html). The main difference in this repository is that the code is (1) up to date as of April 2023 and (2) is compatible with Pytorch XLA on TPU(s).

## How to run
1. Setup TPU. Please refer to the [tpu docs](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm). Make sure Pytorch is setup

2. Install requirements for DQN from [lightning docs](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html).

3. Run `python3 main_tpu.py`
