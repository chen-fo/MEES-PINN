""" base pinn network definition"""

import jax
import jax.numpy as jnp
from typing import Dict, Callable, Tuple
from flax import linen as nn
from abc import abstractmethod

# nodes default value
nodes = 8


class BaseNN(nn.Module):
    """通用 MLP，支持可变输入维度和输出通道数"""
    input_dim: int  # 传入坐标/特征的维度，例如 2 表示 (x,t)，3 表示 (x,y,t)
    output_dim: int = 1  # 网络最终输出通道数，标量场用 1，多场耦合问题可 >1
    width: int = nodes
    depth: int = 4

    @nn.compact
    def __call__(self, inputs):
        """
        inputs: shape (N, input_dim)
        return: shape (N, output_dim)
        """
        h = inputs
        for _ in range(self.depth - 1):
            h = nn.tanh(nn.Dense(self.width, kernel_init=jax.nn.initializers.glorot_uniform())(h))
        return nn.Dense(self.output_dim)(h)

    @abstractmethod
    def derivatives(self, params, X: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        抽象方法
        """

