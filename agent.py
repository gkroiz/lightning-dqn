from typing import Tuple

import numpy as np
import gym
import torch
from torch import nn

from utils import ReplayBuffer
from utils import Experience

import torch_xla.core.xla_model as xm

class Agent:
    """Base Agent class handling the interaction with the environment.

    >>> env = gym.make("CartPole-v1")
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.Agent object at ...>
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state, _ = self.env.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state, _ = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state], device=xm.xla_device())

            if device not in ["cpu", "tpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = "tpu") -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, terminated, truncated , info = self.env.step(action)
        done = truncated or terminated

        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done
